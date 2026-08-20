#!/usr/bin/env python3
"""
Apply the same post-processing used for GatedLSTM to OoBNet and IODA raw
prediction CSVs, saving smoothed versions ready for metric computation or
the demo viewer.

Usage:
    python eval/smooth_baseline_preds.py \
        --oobnet-src /path/to/oobnet_raw_preds \
        --oobnet-dst /path/to/oobnet_smoothed_preds \
        --ioda-src   /path/to/ioda_raw_preds \
        --ioda-dst   /path/to/ioda_smoothed_preds \
        --smooth-window 25 --min-oob-frames 10 --threshold 0.5
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def sliding_majority_vote(flags, window=25):
    n, half = len(flags), window // 2
    return [int(sum(flags[max(0, i-half):min(n, i+half+1)]) >
                (min(n, i+half+1) - max(0, i-half)) / 2)
            for i in range(n)]


def remove_short_segments(flags, min_frames=10):
    flags = list(flags)
    i = 0
    while i < len(flags):
        if flags[i] == 1:
            j = i
            while j < len(flags) and flags[j] == 1:
                j += 1
            if (j - i) < min_frames:
                for k in range(i, j):
                    flags[k] = 0
            i = j
        else:
            i += 1
    return flags


def smooth_pred_dir(pred_dir, out_dir, threshold, smooth_window, min_oob_frames):
    pred_dir = Path(pred_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(pred_dir.glob("*_pred.csv"))
    if not csv_files:
        print(f"  [SKIP] No *_pred.csv in {pred_dir}")
        return

    print(f"\n  Source : {pred_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Files  : {len(csv_files)}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        # Derive OoB probability from label + confidence columns
        prob_oob = np.where(
            df["label"].values == "out-of-body",
            df["confidence"].values,
            1.0 - df["confidence"].values,
        )

        # Apply smoothing
        preds_raw = (prob_oob >= threshold).astype(int)
        preds_s   = sliding_majority_vote(preds_raw.tolist(), window=smooth_window)
        preds_s   = remove_short_segments(preds_s, min_frames=min_oob_frames)
        preds_s   = np.array(preds_s)

        # Build output dataframe keeping all original columns, replacing label
        out_df = df.copy()
        label_inv = {0: "in-body", 1: "out-of-body"}
        out_df["label"]      = [label_inv[p] for p in preds_s]
        out_df["confidence"] = [float(prob_oob[i]) if preds_s[i] == 1
                                else float(1.0 - prob_oob[i])
                                for i in range(len(preds_s))]

        out_path = out_dir / csv_path.name
        out_df.to_csv(out_path, index=False)

        n_oob_raw    = preds_raw.sum()
        n_oob_smooth = preds_s.sum()
        print(f"    {csv_path.name}: OoB frames raw={n_oob_raw} → smoothed={n_oob_smooth}")

    print(f"  Done → {out_dir}")


def main(args):
    models = {
        "oobnet": {"src": args.oobnet_src, "dst": args.oobnet_dst},
        "ioda":   {"src": args.ioda_src,   "dst": args.ioda_dst},
    }

    for name, paths in models.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")
        smooth_pred_dir(
            paths["src"], paths["dst"],
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            min_oob_frames=args.min_oob_frames,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oobnet-src", type=str, required=True)
    parser.add_argument("--oobnet-dst", type=str, required=True)
    parser.add_argument("--ioda-src",   type=str, required=True)
    parser.add_argument("--ioda-dst",   type=str, required=True)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--smooth-window",  type=int,   default=25)
    parser.add_argument("--min-oob-frames", type=int,   default=10)
    main(parser.parse_args())
