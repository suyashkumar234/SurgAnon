#!/usr/bin/env python3
"""
Stage-2: GatedTextFusion + LSTM temporal head.

Per-frame:  SurgVLP (layer4 unfrozen) → img_emb (768)
            13 IB + 8 OB prompts (frozen) → gated fusion via learned gates
            ib_gate: img_feat → softmax → (B, 13)
            ob_gate: img_feat → softmax → (B, 8)
            ib_fused = gate @ text_proj(ib_prompts)  (B, 768)
            ob_fused = gate @ text_proj(ob_prompts)  (B, 768)
            concat [img | ib_fused | ob_fused] → 2304
            LayerNorm → Linear(2304, 512) → GELU  →  LSTM input (512)

Temporal:   LSTM(512, hidden=512, layers=2, batch_first=True)
            → Linear(512, 2)  per timestep

Training:   Videos processed one-at-a-time in temporal order.
            Hidden state carried across windows within a video, reset between
            videos. Truncated BPTT (detach state between windows).
            This matches inference exactly.
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
from train_gated import IB_PROMPTS, OB_PROMPTS, compute_multi_text_embeddings


# ── Sequence dataset (single video, in temporal order) ────────────────────────

class VideoSequenceDataset(Dataset):
    """Sequential non-overlapping windows from ONE video for training."""
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
        # Pad last window if shorter than seq_len
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1])
            labels.append(labels[-1])
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


# ── Small dataset for alignment diagnostics only ──────────────────────────────

class SequenceDataset(Dataset):
    """Multi-video sequence dataset — used only for alignment metrics."""
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

class GatedTextFusionLSTMClassifier(nn.Module):
    def __init__(self, backbone, n_ib=13, n_ob=8, hidden=512, num_layers=2):
        super().__init__()
        self.backbone   = backbone
        dim             = 768
        self.text_proj  = nn.Linear(dim, dim)
        self.ib_gate    = nn.Linear(dim, n_ib)
        self.ob_gate    = nn.Linear(dim, n_ob)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
        )
        self.lstm     = nn.LSTM(hidden, hidden, num_layers=num_layers,
                                batch_first=True,
                                dropout=0.1 if num_layers > 1 else 0.0)
        self.cls_head = nn.Linear(hidden, 2)

    def get_img_features(self, imgs):
        return self.backbone(inputs_img=imgs, mode="video")["img_emb"]

    def gate_text(self, img_feat, ib_prompts, ob_prompts):
        """img_feat: (N, 768) → ib_fused (N, 768), ob_fused (N, 768)."""
        ib_proj = self.text_proj(ib_prompts)                     # (N_ib, 768)
        ob_proj = self.text_proj(ob_prompts)                     # (N_ob, 768)
        ib_w    = torch.softmax(self.ib_gate(img_feat), dim=-1)  # (N, N_ib)
        ob_w    = torch.softmax(self.ob_gate(img_feat), dim=-1)  # (N, N_ob)
        return ib_w @ ib_proj, ob_w @ ob_proj                    # (N, 768) each

    def classify_sequence(self, img_seq, ib_prompts, ob_prompts, hidden=None):
        """img_seq: (1, T, 768) → logits (1, T, 2), hidden, ib_fused, ob_fused."""
        B, T, D  = img_seq.shape
        img_flat = img_seq.view(B*T, D)
        ib_fused, ob_fused = self.gate_text(img_flat, ib_prompts, ob_prompts)
        ib_fused = ib_fused.view(B, T, D)
        ob_fused = ob_fused.view(B, T, D)
        fused    = torch.cat([img_seq, ib_fused, ob_fused], dim=-1)  # (B, T, 2304)
        x        = self.input_proj(fused)                             # (B, T, hidden)
        out, h   = self.lstm(x, hidden)
        return self.cls_head(out), h, ib_fused, ob_fused             # (B, T, 2)

    def classify(self, img_feat, ib_prompts, ob_prompts):
        """Single-frame wrapper for alignment metrics."""
        ib_fused, ob_fused = self.gate_text(img_feat, ib_prompts, ob_prompts)
        fused  = torch.cat([img_feat, ib_fused, ob_fused], dim=-1).unsqueeze(1)
        x      = self.input_proj(fused)
        out, _ = self.lstm(x)
        return self.cls_head(out.squeeze(1)), ib_fused, ob_fused

    def set_train_mode(self, unfreeze_backbone=True):
        self.train()
        self.backbone.eval()
        if unfreeze_backbone:
            for name, module in self.backbone.named_modules():
                if "layer4" in name:
                    module.train()


# ── Alignment diagnostics ─────────────────────────────────────────────────────

@torch.no_grad()
def print_alignment_metrics(model, sample_feats, sample_labels, ib_prompts, ob_prompts):
    feats_n      = F.normalize(sample_feats, dim=-1)
    ib_imgs      = feats_n[sample_labels == 0]
    ob_imgs      = feats_n[sample_labels == 1]
    ib_raw       = F.normalize(ib_prompts.mean(0), dim=-1)
    ob_raw       = F.normalize(ob_prompts.mean(0), dim=-1)
    ib_mean_feat = sample_feats[sample_labels == 0].mean(0, keepdim=True)
    ob_mean_feat = sample_feats[sample_labels == 1].mean(0, keepdim=True)
    _, ib_fused_ib, _ = model.classify(ib_mean_feat, ib_prompts, ob_prompts)
    _, ib_fused_ob, ob_fused_ob = model.classify(ob_mean_feat, ib_prompts, ob_prompts)
    ib_fused_ib  = F.normalize(ib_fused_ib.squeeze(0), dim=-1)
    ob_fused_ob  = F.normalize(ob_fused_ob.squeeze(0), dim=-1)

    def mc(a, b):
        if a.dim() == 1: a = a.unsqueeze(0)
        if b.dim() == 1: b = b.unsqueeze(0)
        return (a @ b.T).mean().item()

    print(f"  ── Mean raw prompts ──────────────────────────────────")
    print(f"  {'cos(mean_ib_prompts, mean_ob_prompts)':<42} {mc(ib_raw, ob_raw):+.4f}")
    print(f"  ── Image features ────────────────────────────────────")
    print(f"  {'cos(ib_img,  ob_img)':<42} {mc(ib_imgs, ob_imgs):+.4f}  (want LOW)")
    print(f"  ── Mean raw prompts ↔ Image ──────────────────────────")
    print(f"  {'cos(ib_prompt, ib_img)':<42} {mc(ib_raw, ib_imgs):+.4f}  (want HIGH ↑)")
    print(f"  {'cos(ob_prompt, ob_img)':<42} {mc(ob_raw, ob_imgs):+.4f}  (want HIGH ↑)")
    print(f"  {'cos(ib_prompt, ob_img)':<42} {mc(ib_raw, ob_imgs):+.4f}  (want LOW  ↓)")
    print(f"  {'cos(ob_prompt, ib_img)':<42} {mc(ob_raw, ib_imgs):+.4f}  (want LOW  ↓)")
    print(f"  ── Gated fused text ↔ Image ──────────────────────────")
    print(f"  {'cos(ib_fused[ib_query], ib_img)':<42} {mc(ib_fused_ib, ib_imgs):+.4f}  (want HIGH ↑)")
    print(f"  {'cos(ob_fused[ob_query], ob_img)':<42} {mc(ob_fused_ob, ob_imgs):+.4f}  (want HIGH ↑)")
    print(f"  {'cos(ib_fused[ib_query], ob_img)':<42} {mc(ib_fused_ib, ob_imgs):+.4f}  (want LOW  ↓)")
    print(f"  {'cos(ob_fused[ob_query], ib_img)':<42} {mc(ob_fused_ob, ib_imgs):+.4f}  (want LOW  ↓)")


# ── Gate weight diagnostics ──────────────────────────────────────────────────

@torch.no_grad()
def print_gate_diagnostics(model, sample_feats, sample_labels, n_ib, n_ob):
    model.eval()
    ib_feat = sample_feats[sample_labels == 0]
    ob_feat = sample_feats[sample_labels == 1]

    def gate_stats(feat, gate_module, label, n_prompts):
        if len(feat) == 0:
            print(f"  {label}: no samples"); return
        w        = torch.softmax(gate_module(feat), dim=-1)
        mean_w   = w.mean(0)
        entropy  = -(w * w.clamp(min=1e-9).log()).sum(-1).mean().item()
        max_ent  = torch.log(torch.tensor(float(n_prompts))).item()
        collapse = 1.0 - entropy / max_ent
        top1     = w.argmax(-1)
        mode_idx = int(top1.mode().values.item())
        mode_pct = (top1 == mode_idx).float().mean().item() * 100
        w_str    = "  ".join(f"p{i}:{v:.3f}" for i, v in enumerate(mean_w.tolist()))
        print(f"  {label}  entropy={entropy:.3f}/{max_ent:.3f}nats  "
              f"collapse={collapse:.3f}  dominant=p{mode_idx}({mode_pct:.0f}%)")
        print(f"    weights: {w_str}")

    print("  ── Gate diagnostics ──────────────────────────────────")
    gate_stats(ib_feat, model.ib_gate, "ib_gate (IB frames )", n_ib)
    gate_stats(ob_feat, model.ob_gate, "ob_gate (OB frames )", n_ob)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_videos_df, preprocess, ib_prompts, ob_prompts,
             device, args, pred_dir=None):
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
            logits, hidden, *_ = model.classify_sequence(
                feat.unsqueeze(0), ib_prompts, ob_prompts, hidden)
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
                    "reason":        "stage2_gated_lstm",
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

    print("Computing multi-prompt text embeddings …")
    ib_prompts, ob_prompts = compute_multi_text_embeddings(labeler.model, device)

    for p in backbone.parameters():
        p.requires_grad = False
    unfrozen = 0
    for name, p in backbone.named_parameters():
        if "layer4" in name:
            p.requires_grad = True
            unfrozen += p.numel()
    print(f"  Unfrozen (layer4): {unfrozen:,} params")

    model = GatedTextFusionLSTMClassifier(backbone,
                                          n_ib=len(IB_PROMPTS), n_ob=len(OB_PROMPTS),
                                          hidden=args.hidden,
                                          num_layers=args.num_layers).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable  : {total:,} params")

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Warm-start: {args.init_checkpoint}  epoch={ckpt['epoch']}  F1={ckpt['f1_smooth']:.4f}")

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

    if args.oversample_labeled > 1:
        labeled_vids = list(train_video_dfs.keys())
        for vid in labeled_vids:
            for i in range(1, args.oversample_labeled):
                train_video_dfs[f"{vid}_copy{i}"] = train_video_dfs[vid]
        print(f"  Oversampled {len(labeled_vids)} labeled videos ×{args.oversample_labeled} "
              f"→ {len(labeled_vids) * args.oversample_labeled} entries")

    if args.pseudo_dir:
        pseudo_csvs = sorted(Path(args.pseudo_dir).glob("*_labels.csv"))
        print(f"\nLoading pseudo-label CSVs from {args.pseudo_dir} …")
        for csv_path in pseudo_csvs:
            vid = csv_path.stem.replace("_labels", "")
            if vid in train_video_dfs or vid in TEST_VIDEOS:
                continue
            df = pd.read_csv(csv_path)
            df = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
            if len(df) == 0:
                continue
            train_video_dfs[vid] = df
            n_pos = (df["label"] == "out-of-body").sum()
            print(f"  [pseudo] {vid}: {len(df)} frames  IB={len(df)-n_pos}  OoB={n_pos}")

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
    head_params   = (list(model.text_proj.parameters()) +
                     list(model.ib_gate.parameters()) +
                     list(model.ob_gate.parameters()) +
                     list(model.input_proj.parameters()) +
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
    print_alignment_metrics(model, sf.to(device), sl.to(device), ib_prompts, ob_prompts)
    print("\nGate diagnostics:")
    print_gate_diagnostics(model, sf.to(device), sl.to(device), len(IB_PROMPTS), len(OB_PROMPTS))
    print("\nValidation:")
    evaluate(model, test_videos_df, preprocess, ib_prompts, ob_prompts, device, args)

    for epoch in range(1, args.epochs + 1):
        unfreeze_backbone = (epoch >= args.unfreeze_epoch)
        model.set_train_mode(unfreeze_backbone=unfreeze_backbone)
        for pg in optimizer.param_groups[:1]:
            pg["lr"] = args.lr_backbone if unfreeze_backbone else 0.0
        if epoch == args.unfreeze_epoch:
            print(f"  [Epoch {epoch}] Unfreezing backbone layer4")
        total_focal = total_align = total_n = total_correct = 0

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
                T = imgs_seq.shape[1]
                imgs_flat   = imgs_seq[0].to(device)
                labels_flat = labels_seq[0].to(device)

                feat_flat = model.get_img_features(imgs_flat)
                feat_seq  = feat_flat.unsqueeze(0)

                logits_seq, h, ib_fused_seq, ob_fused_seq = model.classify_sequence(
                    feat_seq, ib_prompts, ob_prompts, h)
                h = tuple(hh.detach() for hh in h)

                logits_flat   = logits_seq[0]
                ib_fused_flat = ib_fused_seq[0]
                ob_fused_flat = ob_fused_seq[0]

                loss_focal = focal(logits_flat, labels_flat)

                ib_mask = (labels_flat == 0)
                ob_mask = (labels_flat == 1)
                loss_align = torch.tensor(0.0, device=device)
                n_terms = 0
                if ib_mask.sum() > 0:
                    ib_feat_n  = F.normalize(feat_flat[ib_mask].detach(), dim=-1)
                    ib_fused_n = F.normalize(ib_fused_flat[ib_mask], dim=-1)
                    loss_align = loss_align + (1 - F.cosine_similarity(ib_fused_n, ib_feat_n)).mean()
                    n_terms += 1
                if ob_mask.sum() > 0:
                    ob_feat_n  = F.normalize(feat_flat[ob_mask].detach(), dim=-1)
                    ob_fused_n = F.normalize(ob_fused_flat[ob_mask], dim=-1)
                    loss_align = loss_align + (1 - F.cosine_similarity(ob_fused_n, ob_feat_n)).mean()
                    n_terms += 1
                if n_terms > 0:
                    loss_align = loss_align / n_terms

                loss = (loss_focal + args.lambda_align * loss_align) / args.grad_accum
                loss.backward()

                total_focal   += loss_focal.item() * T
                total_align   += loss_align.item() * T
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
              f"align={total_align/total_n:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"lr_backbone={scheduler.get_last_lr()[0]:.2e}  "
              f"lr_heads={scheduler.get_last_lr()[1]:.2e}")
        print(f"{'='*70}")

        sf, sl = get_sample_feats()
        print("\nAlignment metrics:")
        print_alignment_metrics(model, sf.to(device), sl.to(device), ib_prompts, ob_prompts)
        print("\nGate diagnostics:")
        print_gate_diagnostics(model, sf.to(device), sl.to(device), len(IB_PROMPTS), len(OB_PROMPTS))

        print("\nValidation:")
        avg_f1_raw, avg_f1_smooth, avg_auroc, avg_ap, avg_val_loss, avg_val_acc = evaluate(
            model, test_videos_df, preprocess, ib_prompts, ob_prompts, device, args)

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
    parser.add_argument("--epochs",             type=int,   default=15)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--lr-backbone",        type=float, default=1e-4)
    parser.add_argument("--seq-len",            type=int,   default=64)
    parser.add_argument("--stride",             type=int,   default=32)
    parser.add_argument("--grad-accum",         type=int,   default=4,
                        help="Accumulate gradients over N windows before stepping")
    parser.add_argument("--hidden",             type=int,   default=512)
    parser.add_argument("--num-layers",         type=int,   default=2)
    parser.add_argument("--lambda-align",       type=float, default=0.5)
    parser.add_argument("--smooth-window",      type=int,   default=25)
    parser.add_argument("--min-oob-frames",     type=int,   default=10)
    parser.add_argument("--pseudo-dir",         type=str,   default=None)
    parser.add_argument("--oversample-labeled", type=int,   default=1,
                        help="Repeat each labeled video N times to balance against pseudo data")
    parser.add_argument("--init-checkpoint",    type=str,   default=None)
    parser.add_argument("--unfreeze-epoch",     type=int,   default=4,
                        help="Epoch at which to unfreeze backbone layer4")
    parser.add_argument("--ckpt-dir",           type=str,   default="checkpoints")
    parser.add_argument("--ckpt-name",          type=str,   default="best_gated_lstm.pt")
    args = parser.parse_args()
    main(args)
