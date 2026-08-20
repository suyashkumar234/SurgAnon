#!/usr/bin/env python3
"""
Re-generate pseudo labels for unlabeled videos using the trained GatedTextFusion LSTM.

Runs stateful LSTM inference video by video, saves prediction CSVs, and prints
a confidence ranking so you can pick the top N videos for semi-supervised training.

Usage:
    python gen_pseudo_labels.py \
        --checkpoint         checkpoints/best_gated_lstm_last.pt \
        --pseudo-input-dir   /path/to/stage1/pseudo \
        --pseudo-output-dir  /path/to/stage2/pseudo_v2 \
        --top-n              15
"""

import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from vllm_labeler import SurgVLPLabeler
from train_gated import IB_PROMPTS, OB_PROMPTS, compute_multi_text_embeddings
from train_gated_lstm import GatedTextFusionLSTMClassifier
from utils import TRAIN_VIDEOS, TEST_VIDEOS, sliding_majority_vote, remove_short_segments


class FramePathDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df.sort_values("frame_idx").reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.preprocess(Image.open(row["frame_path"]).convert("RGB"))
        return img, idx


@torch.no_grad()
def infer_video(model, df, preprocess, ib_prompts, ob_prompts, device, batch_size):
    """Stateful LSTM inference on one video. Returns OoB probability array (N,)."""
    ds     = FramePathDataset(df, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_probs = []
    h = None
    model.eval()
    for imgs, _ in loader:
        imgs     = imgs.to(device)
        feat     = model.get_img_features(imgs)
        feat_seq = feat.unsqueeze(0)
        logits_seq, h, _, _ = model.classify_sequence(feat_seq, ib_prompts, ob_prompts, h)
        h     = tuple(hh.detach() for hh in h)
        probs = F.softmax(logits_seq[0], dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SurgVLP …")
    labeler    = SurgVLPLabeler(device=device)
    backbone   = labeler.model
    preprocess = labeler.preprocess

    print("Computing multi-prompt text embeddings …")
    ib_prompts, ob_prompts = compute_multi_text_embeddings(labeler.model, device)

    for p in backbone.parameters():
        p.requires_grad = False

    model = GatedTextFusionLSTMClassifier(
        backbone, n_ib=len(IB_PROMPTS), n_ob=len(OB_PROMPTS),
        hidden=args.hidden, num_layers=args.num_layers).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"  epoch={ckpt['epoch']}  F1_smooth={ckpt['f1_smooth']:.4f}")

    skip      = set(TRAIN_VIDEOS) | set(TEST_VIDEOS)
    input_dir = Path(args.pseudo_input_dir)
    csv_paths = sorted(input_dir.glob("*_labels.csv"))
    csv_paths = [p for p in csv_paths
                 if p.stem.replace("_labels", "") not in skip]
    print(f"\nFound {len(csv_paths)} videos to re-label (train/test videos skipped)\n")

    label_map = {0: "in-body", 1: "out-of-body"}
    summary   = []
    results   = {}

    for i, csv_path in enumerate(csv_paths, 1):
        vid = csv_path.stem.replace("_labels", "")
        df  = pd.read_csv(csv_path)
        df  = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
        if len(df) == 0:
            print(f"[{i:02d}/{len(csv_paths)}] SKIP {vid} (no frames)")
            continue

        print(f"[{i:02d}/{len(csv_paths)}] {vid}  ({len(df)} frames) …", flush=True)
        probs = infer_video(model, df, preprocess, ib_prompts, ob_prompts, device, args.batch_size)

        preds_raw = (probs >= 0.5).astype(int)
        preds_s   = sliding_majority_vote(preds_raw.tolist(), window=args.smooth_window)
        preds_s   = remove_short_segments(preds_s, min_frames=args.min_oob_frames)
        preds_s   = np.array(preds_s)

        df = df.sort_values("frame_idx").reset_index(drop=True)
        df["label"]      = [label_map[int(p)] for p in preds_s]
        df["confidence"] = [float(probs[j]) if preds_s[j] == 1 else float(1 - probs[j])
                            for j in range(len(probs))]
        df["reason"]     = "stage2_gated_lstm"

        avg_conf = float(np.mean(np.maximum(probs, 1 - probs)))
        n_oob    = int(preds_s.sum())
        n_ib     = len(preds_s) - n_oob

        print(f"         done  IB={n_ib}  OoB={n_oob}  avg_conf={avg_conf:.4f}", flush=True)

        summary.append({
            "video":    vid,
            "n_frames": len(df),
            "n_ib":     n_ib,
            "n_oob":    n_oob,
            "oob_pct":  100.0 * n_oob / max(len(preds_s), 1),
            "avg_conf": avg_conf,
        })
        results[vid] = df

    summary_df = pd.DataFrame(summary).sort_values("avg_conf", ascending=False).reset_index(drop=True)

    print(f"\n{'='*80}")
    print(f"Video confidence ranking  (top = cleanest pseudo labels)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Video':<42} {'Frames':>7} {'IB':>7} {'OoB':>7} {'OoB%':>6} {'AvgConf':>8}")
    print(f"{'-'*80}")
    for rank, row in enumerate(summary_df.itertuples(), 1):
        marker = " ←" if args.top_n and rank <= args.top_n else ""
        print(f"{rank:<5} {row.video:<42} {row.n_frames:>7} {row.n_ib:>7} {row.n_oob:>7} "
              f"{row.oob_pct:>6.1f}% {row.avg_conf:>8.4f}{marker}")

    output_dir = Path(args.pseudo_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_save = summary_df["video"].tolist()
    if args.top_n:
        to_save = to_save[:args.top_n]
        print(f"\nSaving top {args.top_n} videos (avg_conf ≥ {summary_df.iloc[args.top_n-1]['avg_conf']:.4f}) → {output_dir}")
    else:
        print(f"\nSaving all {len(to_save)} videos → {output_dir}")

    for vid in to_save:
        out_path = output_dir / f"{vid}_labels.csv"
        results[vid].to_csv(out_path, index=False)

    summary_df.to_csv(output_dir / "confidence_ranking.csv", index=False)
    print(f"Saved ranking → {output_dir / 'confidence_ranking.csv'}")
    print(f"\nDone. Use --pseudo-dir {output_dir} in train_gated_lstm_semisup.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        type=str, required=True)
    parser.add_argument("--pseudo-input-dir",  type=str, required=True,
                        help="Dir with stage1 pseudo-label CSVs (has frame_path column)")
    parser.add_argument("--pseudo-output-dir", type=str, required=True,
                        help="Dir to write re-labeled CSVs")
    parser.add_argument("--top-n",             type=int, default=None,
                        help="Only save the top-N most confident videos (default: save all)")
    parser.add_argument("--batch-size",        type=int, default=128)
    parser.add_argument("--hidden",            type=int, default=512)
    parser.add_argument("--num-layers",        type=int, default=2)
    parser.add_argument("--smooth-window",     type=int, default=25)
    parser.add_argument("--min-oob-frames",    type=int, default=10)
    args = parser.parse_args()
    main(args)
