#!/usr/bin/env python3
"""
Standalone eval for GatedTextFusion MLP.
Loads a checkpoint, runs inference on all test videos,
saves per-video *_pred.csv files, and prints smoothed metrics.

Usage:
    python eval_gated_mlp.py \
        --checkpoint checkpoints/best_gated.pt \
        --pred-dir   predictions_gated \
        --batch-size 256 --smooth-window 25 --min-oob-frames 10
"""

import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from vllm_labeler import SurgVLPLabeler
from train_gated import GatedTextFusionClassifier, IB_PROMPTS, OB_PROMPTS, compute_multi_text_embeddings
from utils import FrameDataset, sliding_majority_vote, remove_short_segments, TEST_VIDEOS


@torch.no_grad()
def evaluate(model, test_videos_df, preprocess, ib_prompts, ob_prompts, device, args, pred_dir=None):
    model.eval()
    rows_raw, rows_smooth, aurocs, aps = [], [], [], []

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

        for imgs, _ in loader:
            imgs  = imgs.to(device)
            feat  = model.get_img_features(imgs)
            logits, _, _ = model.classify(feat, ib_prompts, ob_prompts)
            all_probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())

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
                    "reason":        "stage2_gated_mlp",
                    "sampled":       True,
                })
            out_csv = Path(pred_dir) / f"{vid}_pred.csv"
            pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
            print(f"  {'':32}  [CSV] → {out_csv}")

    avg_raw    = float(np.mean(rows_raw))    if rows_raw    else float("nan")
    avg_smooth = float(np.mean(rows_smooth)) if rows_smooth else float("nan")
    avg_auroc  = float(np.mean(aurocs))      if aurocs      else float("nan")
    avg_ap     = float(np.mean(aps))         if aps         else float("nan")
    print(f"  {'AVERAGE':<32} {'':>5} "
          f"{'':>6} {'':>7} {'':>6} {'':>6} "
          f"{'':>7} {'':>6} "
          f"{avg_raw:>7.3f} {avg_smooth:>10.3f} "
          f"{avg_auroc:>7.3f} {avg_ap:>7.3f}")


def main(args):
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

    model = GatedTextFusionClassifier(backbone,
                                      n_ib=len(IB_PROMPTS),
                                      n_ob=len(OB_PROMPTS)).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"  epoch={ckpt['epoch']}  F1_smooth={ckpt['f1_smooth']:.4f}  F1_raw={ckpt.get('f1_raw', float('nan')):.4f}")

    print("\nLoading test label CSVs …")
    test_videos_df = {}
    for vid in TEST_VIDEOS:
        csv_path = gt_dir / f"{vid}_labels.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {vid}"); continue
        df = pd.read_csv(csv_path)
        df = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
        test_videos_df[vid] = df
        n_pos = (df["label"] == "out-of-body").sum()
        print(f"  {vid}: {len(df)} frames  OoB={n_pos}")

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving predictions → {pred_dir}")

    print(f"\nEvaluating (smooth_window={args.smooth_window}, min_oob_frames={args.min_oob_frames}) …")
    evaluate(model, test_videos_df, preprocess, ib_prompts, ob_prompts,
             device, args, pred_dir=str(pred_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--pred-dir",        type=str, required=True)
    parser.add_argument("--batch-size",      type=int, default=256)
    parser.add_argument("--smooth-window",   type=int, default=25)
    parser.add_argument("--min-oob-frames",  type=int, default=10)
    args = parser.parse_args()
    main(args)
