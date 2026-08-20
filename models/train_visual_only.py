#!/usr/bin/env python3
"""
Stage-2 ablation: SurgVLP ResNet50 (layer4 unfrozen) + MLP head, NO text.

Pipeline:
  SurgVLP ResNet50 (layer4 unfrozen) → img_emb (768)
                  ↓
  MLP (768 → LayerNorm → 512 → GELU → Dropout → 2) → FocalLoss
"""

import os, sys, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vllm_labeler import SurgVLPLabeler
from utils import FocalLoss, OoBDataset, FrameDataset, sliding_majority_vote, remove_short_segments, TRAIN_VIDEOS, TEST_VIDEOS


class VisualOnlyClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

    def get_img_features(self, imgs):
        return self.backbone(inputs_img=imgs, mode="video")["img_emb"]

    def classify(self, img_feat):
        return self.cls_head(img_feat)

    def forward(self, imgs):
        feat = self.get_img_features(imgs)
        return self.classify(feat), feat

    def set_train_mode(self):
        self.train()
        self.backbone.eval()
        for name, module in self.backbone.named_modules():
            if "layer4" in name:
                module.train()


@torch.no_grad()
def evaluate(model, test_videos_df, preprocess, device, args, pred_dir=None):
    model.eval()
    rows_raw, rows_smooth, aurocs, aps = [], [], [], []
    focal_fn        = FocalLoss(gamma=2.0)
    val_focal_total = val_n = val_correct = val_total = 0

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

        all_probs = []
        loader = DataLoader(FrameDataset(df, preprocess),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        for imgs, labels_b in loader:
            imgs     = imgs.to(device)
            labels_b = labels_b.to(device)
            logits, _ = model(imgs)
            all_probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            val_focal_total += focal_fn(logits, labels_b).item() * imgs.size(0)
            val_n           += imgs.size(0)
            val_correct     += (logits.argmax(1) == labels_b).sum().item()
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
                pred_lbl = int(preds_s[i]); gt_lbl = int(labels_np[i])
                conf = float(probs[i]) if pred_lbl == 1 else float(1.0 - probs[i])
                row  = df.iloc[i]
                csv_rows.append({"frame_idx": int(row["frame_idx"]),
                                 "timestamp_sec": float(row["timestamp_sec"]),
                                 "frame_path": row["frame_path"],
                                 "label": label_inv[pred_lbl], "gt_label": label_inv[gt_lbl],
                                 "confidence": conf, "correct": pred_lbl == gt_lbl,
                                 "reason": "visual_only_mlp", "sampled": True})
            pd.DataFrame(csv_rows).to_csv(Path(pred_dir) / f"{vid}_pred.csv", index=False)

    avg_raw      = float(np.mean(rows_raw))    if rows_raw    else float("nan")
    avg_smooth   = float(np.mean(rows_smooth)) if rows_smooth else float("nan")
    avg_auroc    = float(np.mean(aurocs))      if aurocs      else float("nan")
    avg_ap       = float(np.mean(aps))         if aps         else float("nan")
    avg_val_loss = val_focal_total / val_n     if val_n > 0   else float("nan")
    avg_val_acc  = val_correct / val_total     if val_total > 0 else float("nan")
    print(f"  {'AVERAGE':<32} {'':>5} {'':>6} {'':>7} {'':>6} {'':>6} {'':>7} {'':>6} "
          f"{avg_raw:>7.3f} {avg_smooth:>10.3f} {avg_auroc:>7.3f} {avg_ap:>7.3f}")
    print(f"  val_focal={avg_val_loss:.4f}  val_acc={avg_val_acc:.4f}")
    return avg_raw, avg_smooth, avg_auroc, avg_ap, avg_val_loss, avg_val_acc


def main(args):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = yaml.safe_load(open(Path(__file__).parent.parent / "src" / "config.yaml"))
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

    model = VisualOnlyClassifier(backbone).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable  : {total:,} params")

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Warm-start: {args.init_checkpoint}")

    print("\nLoading label CSVs …")
    train_csvs = [gt_dir / f"{v}_labels.csv" for v in TRAIN_VIDEOS]
    train_ds   = OoBDataset(train_csvs, preprocess, sample_every=args.sample_every)
    n_oob = (train_ds.df["label"] == "out-of-body").sum()
    n_ib  = len(train_ds.df) - n_oob
    print(f"  Train: {len(train_ds)} frames  IB={n_ib}  OoB={n_oob}")

    test_videos_df = {}
    for vid in TEST_VIDEOS:
        csv = gt_dir / f"{vid}_labels.csv"
        if not csv.exists():
            print(f"  [SKIP] {vid}"); continue
        df = pd.read_csv(csv)
        df = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
        test_videos_df[vid] = df
        n_pos = (df["label"] == "out-of-body").sum()
        print(f"  {vid}: {len(df)} frames  IB={len(df)-n_pos}  OoB={n_pos}")

    w = torch.where(
        torch.tensor(train_ds.df["label"].values == "out-of-body"),
        torch.tensor(1.0 / max(n_oob, 1)),
        torch.tensor(1.0 / max(n_ib,  1)))
    sampler      = WeightedRandomSampler(w, len(w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    focal = FocalLoss(gamma=2.0)
    layer4_params = [p for n, p in model.named_parameters() if "layer4" in n and p.requires_grad]
    head_params   = list(model.cls_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": layer4_params, "lr": args.lr_backbone},
        {"params": head_params,   "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = torch.load(args.init_checkpoint, map_location="cpu")["f1_smooth"] \
              if args.init_checkpoint else 0.0

    for epoch in range(1, args.epochs + 1):
        model.set_train_mode()
        total_focal = total_n = total_correct = 0

        for imgs, labels_b in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False):
            imgs     = imgs.to(device)
            labels_b = labels_b.to(device)
            B        = imgs.shape[0]
            logits, _ = model(imgs)
            loss = focal(logits, labels_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_focal   += loss.item() * B
            total_n       += B
            total_correct += (logits.argmax(1) == labels_b).sum().item()

        scheduler.step()
        train_acc = total_correct / total_n
        print(f"\n{'='*70}")
        print(f"Epoch {epoch:02d}/{args.epochs}  focal={total_focal/total_n:.4f}  train_acc={train_acc:.4f}")
        print(f"{'='*70}")

        avg_f1_raw, avg_f1_smooth, avg_auroc, avg_ap, avg_val_loss, avg_val_acc = evaluate(
            model, test_videos_df, preprocess, device, args)

        marker = " ← best" if avg_f1_smooth > best_f1 else ""
        print(f"\n  Epoch {epoch:02d}  F1_raw={avg_f1_raw:.4f}  F1_smooth={avg_f1_smooth:.4f}{marker}")

        if avg_f1_smooth > best_f1:
            best_f1 = avg_f1_smooth
            torch.save({"epoch": epoch, "f1_smooth": avg_f1_smooth, "f1_raw": avg_f1_raw,
                        "auroc": avg_auroc, "ap": avg_ap, "model_state": model.state_dict()},
                       str(ckpt_dir / args.ckpt_name))
            print(f"  Saved best → {ckpt_dir / args.ckpt_name}")

    last_name = args.ckpt_name.replace(".pt", "_last.pt")
    torch.save({"epoch": args.epochs, "f1_smooth": avg_f1_smooth, "f1_raw": avg_f1_raw,
                "auroc": avg_auroc, "ap": avg_ap, "model_state": model.state_dict()},
               str(ckpt_dir / last_name))
    print(f"  Saved last → {ckpt_dir / last_name}")

    if args.pred_dir:
        print(f"\nFinal eval + CSV export → {args.pred_dir}")
        ckpt = torch.load(str(ckpt_dir / args.ckpt_name), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        pred_dir = Path(args.pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        evaluate(model, test_videos_df, preprocess, device, args, pred_dir=pred_dir)

    print(f"\n{'='*70}\nDone.  Best F1 (smooth) = {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lr-backbone",     type=float, default=1e-4)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--sample-every",    type=int,   default=1)
    parser.add_argument("--smooth-window",   type=int,   default=25)
    parser.add_argument("--min-oob-frames",  type=int,   default=10)
    parser.add_argument("--init-checkpoint", type=str,   default=None)
    parser.add_argument("--ckpt-dir",        type=str,   default="checkpoints")
    parser.add_argument("--ckpt-name",       type=str,   default="best_visual_only.pt")
    parser.add_argument("--pred-dir",        type=str,   default=None,
                        help="If set, save per-video prediction CSVs here after training")
    args = parser.parse_args()
    main(args)
