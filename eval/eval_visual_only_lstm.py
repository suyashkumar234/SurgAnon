#!/usr/bin/env python3
"""
Standalone eval for Visual-Only + LSTM.
Loads a checkpoint, runs stateful inference on all test videos,
saves per-video *_pred.csv files, and prints smoothed metrics.

Usage:
    python eval_visual_only_lstm.py \
        --checkpoint checkpoints/best_visual_only_lstm.pt \
        --pred-dir   predictions_visual_only_lstm \
        --seq-len 64 --smooth-window 25 --min-oob-frames 10
"""

import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, yaml
import torch
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from vllm_labeler import SurgVLPLabeler
from train_visual_only_lstm import VisualOnlyLSTMClassifier, evaluate
from utils import TEST_VIDEOS


def main(args):
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

    model = VisualOnlyLSTMClassifier(backbone,
                                     hidden=args.hidden,
                                     num_layers=args.num_layers).to(device)

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
    evaluate(model, test_videos_df, preprocess, device, args, pred_dir=str(pred_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--pred-dir",        type=str, required=True)
    parser.add_argument("--seq-len",         type=int, default=64)
    parser.add_argument("--hidden",          type=int, default=512)
    parser.add_argument("--num-layers",      type=int, default=2)
    parser.add_argument("--smooth-window",   type=int, default=25)
    parser.add_argument("--min-oob-frames",  type=int, default=10)
    args = parser.parse_args()
    main(args)
