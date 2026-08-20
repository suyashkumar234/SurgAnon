#!/usr/bin/env python3
"""
Stage-2 ablation: Visual-Only + LSTM temporal head.  NO text.

Per-frame:  SurgVLP (layer4 frozen) → img_emb (768)
Temporal:   LayerNorm(768) → Linear(768, 512) → GELU → LSTM(512, hidden=512, layers=2)
            → Linear(512, 2) per timestep

Warm-starts from best_visual_only.pt (MLP head weights discarded via strict=False).
Backbone stays frozen throughout (unfreeze-epoch=999).
This is the missing cell in the 2x2 ablation table (text=no, temporal=yes).
"""

import os, sys, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vllm_labeler import SurgVLPLabeler
from utils import (
    FocalLoss, FrameDataset, sliding_majority_vote, remove_short_segments,
    TRAIN_VIDEOS, TEST_VIDEOS,
)


# ── Sequence dataset (single video, in temporal order) ────────────────────────

class VideoSequenceDataset(Dataset):
    """Sequential overlapping windows from ONE video for stateful training."""
    def __init__(self, df, preprocess, seq_len=64, stride=32):
        self.preprocess = preprocess
        self.seq_len    = seq_len
        self.df         = df.sort_values("frame_idx").reset_index(drop=True)
        n = len(self.df)
        self.windows = []
        for start in range(0, max(1, n - seq_len + 1), stride):
            self.windows.append(start)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end   = min(start + self.seq_len, len(self.df))
        chunk = self.df.iloc[start:end]
        imgs, labels = [], []
        for _, row in chunk.iterrows():
            imgs.append(self.preprocess(Image.open(row["frame_path"]).convert("RGB")))
            labels.append(0 if row["label"] == "in-body" else 1)
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1])
            labels.append(labels[-1])
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


class SequenceDataset(Dataset):
    """Multi-video flat sequence dataset — used only for alignment diagnostics."""
    def __init__(self, csv_paths, preprocess, seq_len=64, stride=32):
        self.preprocess = preprocess
        self.seq_len    = seq_len
        self.seqs       = []
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]
        for p in csv_paths:
            p = Path(p)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df = df[df["label"].isin(["in-body", "out-of-body"])]
            df = df.sort_values("frame_idx").reset_index(drop=True)
            n = len(df)
            for start in range(0, n - seq_len + 1, stride):
                self.seqs.append((df, start))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        df, start = self.seqs[idx]
        chunk = df.iloc[start:start + self.seq_len]
        imgs, labels = [], []
        for _, row in chunk.iterrows():
            imgs.append(self.preprocess(Image.open(row["frame_path"]).convert("RGB")))
            labels.append(0 if row["label"] == "in-body" else 1)
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class VisualOnlyLSTMClassifier(nn.Module):
    """SurgVLP backbone (layer4 frozen) → LSTM temporal head.  No text."""
    def __init__(self, backbone, hidden=512, num_layers=2):
        super().__init__()
        self.backbone   = backbone
        dim             = 768
        self.input_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
        )
        self.lstm     = nn.LSTM(hidden, hidden, num_layers=num_layers,
                                batch_first=True,
                                dropout=0.1 if num_layers > 1 else 0.0)
        self.cls_head = nn.Linear(hidden, 2)

    def get_img_features(self, imgs):
        return self.backbone(inputs_img=imgs, mode="video")["img_emb"]  # (B, 768)

    def classify_sequence(self, img_seq, hidden=None):
        """img_seq: (1, T, 768) → logits (1, T, 2), hidden."""
        x        = self.input_proj(img_seq)       # (B, T, hidden)
        out, h   = self.lstm(x, hidden)
        return self.cls_head(out), h              # (B, T, 2)

    def set_train_mode(self, unfreeze_backbone=False):
        self.train()
        self.backbone.eval()
        if unfreeze_backbone:
            for name, module in self.backbone.named_modules():
                if "layer4" in name:
                    module.train()


# ── Alignment diagnostics (image-only) ───────────────────────────────────────

@torch.no_grad()
def print_alignment_metrics(sample_feats, sample_labels):
    feats_n = F.normalize(sample_feats, dim=-1)
    ib_imgs = feats_n[sample_labels == 0]
    ob_imgs = feats_n[sample_labels == 1]

    def mc(a, b):
        if a.dim() == 1: a = a.unsqueeze(0)
        if b.dim() == 1: b = b.unsqueeze(0)
        return (a @ b.T).mean().item()

    print(f"  ── Image features ────────────────────────────────────")
    print(f"  {'cos(ib_img, ob_img)':<38} {mc(ib_imgs, ob_imgs):+.4f}  (want LOW)")
    print(f"  {'cos(ib_img, ib_img)':<38} {mc(ib_imgs, ib_imgs):+.4f}  (want HIGH)")
    print(f"  {'cos(ob_img, ob_img)':<38} {mc(ob_imgs, ob_imgs):+.4f}  (want HIGH)")


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_videos_df, preprocess, device, args, pred_dir=None):
    model.eval()
    rows_raw, rows_smooth, aurocs, aps = [], [], [], []
    focal_fn        = FocalLoss(gamma=2.0)
    val_focal_total = val_n = 0
    val_correct = val_total = 0

    hdr = (f"{'Video':<32} {'Thr':>5} "
           f"{'TP':>6} {'TN':>7} {'FP':>6} {'FN':>6} "
           f"{'Recall':>7} {'Prec':>6} "
           f"{'F1_raw':>7} {'F1_smooth':>10} "
           f"{'AUROC':>7} {'AP':>7}")
    print(f"\n  {hdr}")
    print(f"  {'-'*110}")

    for vid in TEST_VIDEOS:
        if vid not in test_videos_df:
            continue
        df        = test_videos_df[vid]
        labels_np = df["label"].map({"in-body": 0, "out-of-body": 1}).values

        all_probs, hidden = [], None
        loader = DataLoader(FrameDataset(df, preprocess),
                            batch_size=args.seq_len, shuffle=False,
                            num_workers=4, pin_memory=True)

        for imgs, labels_b in loader:
            imgs     = imgs.to(device)
            labels_b = labels_b.to(device)
            feat     = model.get_img_features(imgs)
            logits, hidden = model.classify_sequence(feat.unsqueeze(0), hidden)
            hidden   = tuple(h.detach() for h in hidden)
            logits_flat = logits[0]
            probs    = F.softmax(logits_flat, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            val_focal_total += focal_fn(logits_flat, labels_b).item() * imgs.size(0)
            val_n           += imgs.size(0)
            val_correct     += (logits_flat.argmax(1) == labels_b).sum().item()
            val_total       += imgs.size(0)

        probs = np.concatenate(all_probs)

        def metrics(preds_arr):
            TP = int(((preds_arr == 1) & (labels_np == 1)).sum())
            TN = int(((preds_arr == 0) & (labels_np == 0)).sum())
            FP = int(((preds_arr == 1) & (labels_np == 0)).sum())
            FN = int(((preds_arr == 0) & (labels_np == 1)).sum())
            rec  = TP / max(TP + FN, 1)
            prec = TP / max(TP + FP, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            return TP, TN, FP, FN, rec, prec, f1

        preds_raw = (probs >= 0.5).astype(int)
        TP, TN, FP, FN, rec, prec, f1_raw = metrics(preds_raw)
        rows_raw.append(f1_raw)

        preds_s = sliding_majority_vote(preds_raw.tolist(), window=args.smooth_window)
        preds_s = remove_short_segments(preds_s, min_frames=args.min_oob_frames)
        preds_s = np.array(preds_s)
        TP, TN, FP, FN, rec, prec, f1_smooth = metrics(preds_s)
        rows_smooth.append(f1_smooth)

        n_pos = labels_np.sum(); n_neg = len(labels_np) - n_pos
        if HAS_SKLEARN and n_pos > 0 and n_neg > 0:
            auroc = float(roc_auc_score(labels_np, probs))
            ap    = float(average_precision_score(labels_np, probs))
            aurocs.append(auroc); aps.append(ap)
            auroc_s, ap_s = f"{auroc:.3f}", f"{ap:.3f}"
        else:
            auroc_s = ap_s = "  N/A"

        print(f"  {vid:<32} {0.5:>5.3f} "
              f"{TP:>6} {TN:>7} {FP:>6} {FN:>6} "
              f"{rec:>7.3f} {prec:>6.3f} "
              f"{f1_raw:>7.3f} {f1_smooth:>10.3f} "
              f"{auroc_s:>7} {ap_s:>7}")

        if pred_dir is not None:
            label_inv = {0: "in-body", 1: "out-of-body"}
            csv_rows  = []
            for i in range(len(preds_s)):
                pred_lbl = int(preds_s[i])
                gt_lbl   = int(labels_np[i])
                conf     = float(probs[i]) if pred_lbl == 1 else float(1.0 - probs[i])
                row      = df.iloc[i]
                csv_rows.append({
                    "frame_idx":     int(row["frame_idx"]),
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "frame_path":    row["frame_path"],
                    "label":         label_inv[pred_lbl],
                    "gt_label":      label_inv[gt_lbl],
                    "confidence":    conf,
                    "correct":       pred_lbl == gt_lbl,
                    "reason":        "stage2_visual_only_lstm",
                    "sampled":       True,
                })
            out_csv = Path(pred_dir) / f"{vid}_pred.csv"
            pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
            print(f"  {'':32}  [CSV] → {out_csv}")

    avg_raw      = float(np.mean(rows_raw))    if rows_raw    else float("nan")
    avg_smooth   = float(np.mean(rows_smooth)) if rows_smooth else float("nan")
    avg_auroc    = float(np.mean(aurocs))      if aurocs      else float("nan")
    avg_ap       = float(np.mean(aps))         if aps         else float("nan")
    avg_val_loss = val_focal_total / val_n     if val_n > 0   else float("nan")
    avg_val_acc  = val_correct / val_total     if val_total > 0 else float("nan")
    print(f"  {'AVERAGE':<32} {'':>5} "
          f"{'':>6} {'':>7} {'':>6} {'':>6} "
          f"{'':>7} {'':>6} "
          f"{avg_raw:>7.3f} {avg_smooth:>10.3f} "
          f"{avg_auroc:>7.3f} {avg_ap:>7.3f}")
    print(f"  val_focal={avg_val_loss:.4f}  val_acc={avg_val_acc:.4f}")
    return avg_raw, avg_smooth, avg_auroc, avg_ap, avg_val_loss, avg_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = yaml.safe_load(
        open(Path(__file__).parent.parent / "src" / "config.yaml"))
    gt_dir = Path(config["step1"]["output_dir"])

    print("Loading SurgVLP …")
    labeler    = SurgVLPLabeler(device=device)
    backbone   = labeler.model
    preprocess = labeler.preprocess

    for p in backbone.parameters():
        p.requires_grad = False
    unfrozen = 0
    for name, p in backbone.named_parameters():
        if "layer4" in name:
            p.requires_grad = True
            unfrozen += p.numel()
    print(f"  Unfrozen (layer4): {unfrozen:,} params")

    model = VisualOnlyLSTMClassifier(backbone,
                                     hidden=args.hidden,
                                     num_layers=args.num_layers).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable  : {total:,} params")

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Warm-start: {args.init_checkpoint}  epoch={ckpt['epoch']}  F1={ckpt['f1_smooth']:.4f}")
        if missing:
            print(f"  Missing keys (new LSTM head — expected): {missing[:4]} ...")
        if unexpected:
            print(f"  Unexpected keys (old MLP head — discarded): {unexpected[:4]} ...")

    print("\nLoading label CSVs …")
    train_video_dfs = {}
    for vid in TRAIN_VIDEOS:
        csv_path = gt_dir / f"{vid}_labels.csv"
        if not csv_path.exists():
            print(f"  [SKIP train] {vid}"); continue
        df = pd.read_csv(csv_path)
        df = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
        train_video_dfs[vid] = df
        n_pos = (df["label"] == "out-of-body").sum()
        print(f"  [train] {vid}: {len(df)} frames  IB={len(df)-n_pos}  OoB={n_pos}")

    test_videos_df = {}
    for vid in TEST_VIDEOS:
        csv_path = gt_dir / f"{vid}_labels.csv"
        if not csv_path.exists():
            print(f"  [SKIP test]  {vid}"); continue
        df = pd.read_csv(csv_path)
        df = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
        test_videos_df[vid] = df
        n_pos = (df["label"] == "out-of-body").sum()
        print(f"  [test]  {vid}: {len(df)} frames  IB={len(df)-n_pos}  OoB={n_pos}")

    focal = FocalLoss(gamma=2.0)

    layer4_params = [p for n, p in model.named_parameters()
                     if "layer4" in n and p.requires_grad]
    head_params   = (list(model.input_proj.parameters()) +
                     list(model.lstm.parameters()) +
                     list(model.cls_head.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": layer4_params, "lr": args.lr_backbone},
        {"params": head_params,   "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_name = args.ckpt_name.replace(".pt", "_last.pt")

    best_f1 = torch.load(args.init_checkpoint, map_location="cpu")["f1_smooth"] \
              if args.init_checkpoint else 0.0

    sample_ds = SequenceDataset([gt_dir / f"{v}_labels.csv" for v in TRAIN_VIDEOS],
                                preprocess, seq_len=args.seq_len, stride=args.seq_len)

    def get_sample_feats(n=2000):
        idx    = torch.randperm(len(sample_ds))[:32].tolist()
        subset = torch.utils.data.Subset(sample_ds, idx)
        loader = DataLoader(subset, batch_size=4, shuffle=False,
                            num_workers=4, pin_memory=True)
        feats, lbls = [], []
        with torch.no_grad():
            model.eval()
            for imgs_seq, lbl_seq in loader:
                B, T, C, H, W = imgs_seq.shape
                f = model.get_img_features(imgs_seq.view(B*T, C, H, W).to(device))
                feats.append(f.cpu()); lbls.append(lbl_seq.view(-1))
        return torch.cat(feats)[:n], torch.cat(lbls)[:n]

    print(f"\n{'='*70}")
    print("Epoch 00 — BASELINE")
    print(f"{'='*70}")
    sf, sl = get_sample_feats()
    print("\nAlignment metrics:")
    print_alignment_metrics(sf.to(device), sl.to(device))
    print("\nValidation:")
    evaluate(model, test_videos_df, preprocess, device, args)

    for epoch in range(1, args.epochs + 1):
        unfreeze_backbone = (epoch >= args.unfreeze_epoch)
        model.set_train_mode(unfreeze_backbone=unfreeze_backbone)
        for pg in optimizer.param_groups[:1]:
            pg["lr"] = args.lr_backbone if unfreeze_backbone else 0.0
        if epoch == args.unfreeze_epoch:
            print(f"  [Epoch {epoch}] Unfreezing backbone layer4")
        total_focal = total_n = total_correct = 0

        video_order = list(train_video_dfs.keys())
        random.shuffle(video_order)

        epoch_bar = tqdm(video_order, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False)
        for vid in epoch_bar:
            df     = train_video_dfs[vid]
            vid_ds = VideoSequenceDataset(df, preprocess,
                                          seq_len=args.seq_len, stride=args.stride)
            vid_loader = DataLoader(vid_ds, batch_size=1, shuffle=False,
                                    num_workers=4, pin_memory=True)

            h = None
            optimizer.zero_grad()
            accum = 0

            for imgs_seq, labels_seq in vid_loader:
                T           = imgs_seq.shape[1]
                imgs_flat   = imgs_seq[0].to(device)
                labels_flat = labels_seq[0].to(device)

                feat_flat = model.get_img_features(imgs_flat)
                feat_seq  = feat_flat.unsqueeze(0)

                logits_seq, h = model.classify_sequence(feat_seq, h)
                h = tuple(hh.detach() for hh in h)

                logits_flat = logits_seq[0]
                loss = focal(logits_flat, labels_flat) / args.grad_accum
                loss.backward()

                total_focal   += loss.item() * args.grad_accum * T
                total_n       += T
                total_correct += (logits_flat.argmax(1) == labels_flat).sum().item()

                accum += 1
                if accum % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if accum % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        train_acc = total_correct / total_n
        print(f"\n{'='*70}")
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"focal={total_focal/total_n:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"lr_backbone={scheduler.get_last_lr()[0]:.2e}  "
              f"lr_heads={scheduler.get_last_lr()[1]:.2e}")
        print(f"{'='*70}")

        sf, sl = get_sample_feats()
        print("\nAlignment metrics:")
        print_alignment_metrics(sf.to(device), sl.to(device))

        print("\nValidation:")
        avg_f1_raw, avg_f1_smooth, avg_auroc, avg_ap, avg_val_loss, avg_val_acc = evaluate(
            model, test_videos_df, preprocess, device, args)

        marker = " ← best" if avg_f1_smooth > best_f1 else ""
        print(f"\n  Epoch {epoch:02d}  "
              f"train_focal={total_focal/total_n:.4f}  train_acc={train_acc:.4f}  "
              f"val_focal={avg_val_loss:.4f}  val_acc={avg_val_acc:.4f}  "
              f"F1_raw={avg_f1_raw:.4f}  F1_smooth={avg_f1_smooth:.4f}  "
              f"AUROC={avg_auroc:.4f}{marker}")

        if avg_f1_smooth > best_f1:
            best_f1 = avg_f1_smooth
            torch.save({
                "epoch": epoch, "f1_smooth": avg_f1_smooth, "f1_raw": avg_f1_raw,
                "auroc": avg_auroc, "ap": avg_ap, "model_state": model.state_dict(),
            }, str(ckpt_dir / args.ckpt_name))
            print(f"  Saved best → {ckpt_dir / args.ckpt_name}  (epoch {epoch})")

    torch.save({
        "epoch": args.epochs, "f1_smooth": avg_f1_smooth, "f1_raw": avg_f1_raw,
        "auroc": avg_auroc, "ap": avg_ap, "model_state": model.state_dict(),
    }, str(ckpt_dir / last_ckpt_name))
    print(f"  Saved last → {ckpt_dir / last_ckpt_name}")

    print(f"\n{'='*70}")
    print(f"Done.  Best F1 (smooth) = {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lr-backbone",     type=float, default=1e-4)
    parser.add_argument("--seq-len",         type=int,   default=64)
    parser.add_argument("--stride",          type=int,   default=32)
    parser.add_argument("--grad-accum",      type=int,   default=4)
    parser.add_argument("--hidden",          type=int,   default=512)
    parser.add_argument("--num-layers",      type=int,   default=2)
    parser.add_argument("--smooth-window",   type=int,   default=25)
    parser.add_argument("--min-oob-frames",  type=int,   default=10)
    parser.add_argument("--init-checkpoint", type=str,   default=None)
    parser.add_argument("--unfreeze-epoch",  type=int,   default=999,
                        help="Epoch to unfreeze backbone layer4 (999 = never)")
    parser.add_argument("--ckpt-dir",        type=str,   default="checkpoints")
    parser.add_argument("--ckpt-name",       type=str,   default="best_visual_only_lstm.pt")
    args = parser.parse_args()
    main(args)
