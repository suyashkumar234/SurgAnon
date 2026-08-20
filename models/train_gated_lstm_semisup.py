#!/usr/bin/env python3
"""
Semi-supervised fine-tuning of GatedTextFusion LSTM.

Loads a supervised checkpoint, freezes backbone + text_proj + gates,
and trains only input_proj + LSTM + cls_head on labeled + pseudo-labeled data.
Focal loss handles class imbalance — no oversampling needed.

Usage:
    python train_gated_lstm_semisup.py \
        --init-checkpoint checkpoints/best_gated_lstm_last.pt \
        --pseudo-dir      /path/to/pseudo \
        --epochs 10 --lr 5e-4
"""
import os, sys, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vllm_labeler import SurgVLPLabeler
from utils import FocalLoss, TRAIN_VIDEOS, TEST_VIDEOS
from train_gated import IB_PROMPTS, OB_PROMPTS, compute_multi_text_embeddings
from train_gated_lstm import (
    GatedTextFusionLSTMClassifier, VideoSequenceDataset, SequenceDataset,
    evaluate, print_alignment_metrics, print_gate_diagnostics,
)


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

    model = GatedTextFusionLSTMClassifier(backbone,
                                          n_ib=len(IB_PROMPTS), n_ob=len(OB_PROMPTS),
                                          hidden=args.hidden,
                                          num_layers=args.num_layers).to(device)

    print(f"Loading checkpoint: {args.init_checkpoint}")
    ckpt = torch.load(args.init_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"  epoch={ckpt['epoch']}  F1_smooth={ckpt['f1_smooth']:.4f}")

    trainable = (list(model.text_proj.parameters()) +
                 list(model.ib_gate.parameters()) +
                 list(model.ob_gate.parameters()) +
                 list(model.input_proj.parameters()) +
                 list(model.lstm.parameters()) +
                 list(model.cls_head.parameters()))
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Frozen    (backbone): {sum(p.numel() for p in backbone.parameters()):,} params")
    print(f"  Trainable (text_proj + gates + input_proj + LSTM + cls_head): {n_trainable:,} params")

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
            n_pos = (df["label"] == "out-of-body").sum()
            print(f"  [pseudo] {vid}: {len(df)} frames  IB={len(df)-n_pos}  OoB={n_pos}")
            train_video_dfs[vid] = df

    print(f"\n  Total training videos: {len(train_video_dfs)}")

    test_videos_df = {}
    for vid in TEST_VIDEOS:
        csv_path = gt_dir / f"{vid}_labels.csv"
        if not csv_path.exists():
            print(f"  [SKIP test] {vid}"); continue
        df = pd.read_csv(csv_path)
        df = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
        test_videos_df[vid] = df
        n_pos = (df["label"] == "out-of-body").sum()
        print(f"  [test]  {vid}: {len(df)} frames  IB={len(df)-n_pos}  OoB={n_pos}")

    focal = FocalLoss(gamma=2.0)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.init_checkpoint).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()

        total_focal = total_align = total_n = total_correct = 0

        video_order = list(train_video_dfs.keys())
        random.shuffle(video_order)

        for vid in tqdm(video_order, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False):
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

                with torch.no_grad():
                    feat_flat = model.get_img_features(imgs_flat)
                feat_seq = feat_flat.unsqueeze(0)

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
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if accum % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        train_focal = total_focal / max(total_n, 1)
        train_align = total_align / max(total_n, 1)
        train_acc   = total_correct / max(total_n, 1)
        lr_now      = scheduler.get_last_lr()[0]

        print(f"\n{'='*70}")
        print(f"Epoch {epoch:02d}/{args.epochs}  focal={train_focal:.4f}  "
              f"align={train_align:.4f}  train_acc={train_acc:.4f}  lr={lr_now:.2e}")
        print(f"{'='*70}")

        sf, sl = get_sample_feats()
        print("\nAlignment metrics:")
        print_alignment_metrics(model, sf.to(device), sl.to(device), ib_prompts, ob_prompts)
        print("\nGate diagnostics:")
        print_gate_diagnostics(model, sf.to(device), sl.to(device), len(IB_PROMPTS), len(OB_PROMPTS))

        print(f"\n  Epoch {epoch:02d}  train_focal={train_focal:.4f}  train_acc={train_acc:.4f}")

    print(f"\n{'='*70}")
    print("Final evaluation on test set …")
    print(f"{'='*70}")
    avg_f1_raw, avg_f1_smooth, avg_auroc, avg_ap, _, _ = evaluate(
        model, test_videos_df, preprocess, ib_prompts, ob_prompts, device, args)

    torch.save({
        "epoch": args.epochs, "f1_smooth": avg_f1_smooth, "f1_raw": avg_f1_raw,
        "auroc": avg_auroc, "ap": avg_ap, "model_state": model.state_dict(),
    }, str(ckpt_dir / args.ckpt_name))
    print(f"  Saved → {ckpt_dir / args.ckpt_name}")
    print(f"\n{'='*70}")
    print(f"Done.  F1_smooth={avg_f1_smooth:.4f}  AUROC={avg_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str,   required=True)
    parser.add_argument("--pseudo-dir",      type=str,   default=None)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--lr",              type=float, default=5e-4)
    parser.add_argument("--seq-len",         type=int,   default=64)
    parser.add_argument("--stride",          type=int,   default=32)
    parser.add_argument("--grad-accum",      type=int,   default=4)
    parser.add_argument("--hidden",          type=int,   default=512)
    parser.add_argument("--num-layers",      type=int,   default=2)
    parser.add_argument("--lambda-align",    type=float, default=0.5)
    parser.add_argument("--smooth-window",   type=int,   default=25)
    parser.add_argument("--min-oob-frames",  type=int,   default=10)
    parser.add_argument("--ckpt-name",       type=str,   default="best_gated_lstm_semisup.pt")
    args = parser.parse_args()
    main(args)
