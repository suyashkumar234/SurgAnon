#!/usr/bin/env python3
"""
Stage-2: Gated Multi-Prompt TextFusion OoB detection.

Architecture:
  SurgVLP ResNet50 (layer4 unfrozen) → img_emb (768)
                                           |
  13 IB prompts + 8 OB prompts (frozen)    |
  text_proj (768→768, trainable)           |
                ↓                          |
  ib_gate: img_feat → softmax → (B, 13)   |
  ob_gate: img_feat → softmax → (B, 8)    |
  ib_fused = gate @ text_proj(ib_prompts) |
  ob_fused = gate @ text_proj(ob_prompts) |
                ↓                          ↓
  Concat [img_emb | ib_fused | ob_fused] → 2304
                ↓
  MLP (2304→512→2) → FocalLoss + alignment loss
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
import surgvlp
from utils import OoBDataset, FrameDataset, FocalLoss, sliding_majority_vote, remove_short_segments, TRAIN_VIDEOS, TEST_VIDEOS

IB_PROMPTS = [
    "robotic instruments grasping and cutting fatty tissue inside body cavity",
    "robotic instruments placing surgical mesh inside the patient's abdominal cavity",
    "robotic instruments suturing and stitching tissue inside the patient's abdomen",
    "robotic laparoscopic instruments dissecting connective tissue inside the abdomen",
    "da vinci robotic arm retracting tissue and organs inside the body during surgery",
    "robotic instruments creating smoke while cauterizing tissue inside the patient's abdomen",
    "robotic laparoscopic instruments inserting tissue into a retrieval bag inside the patient's abdomen",
    "robotic instruments clipping blood vessels inside the patient's abdominal cavity",
    "robotic laparoscopic instruments entering through port inside the patient's abdomen",
    "robotic instruments working with fluid inside the patient's body during surgery",
    "robotic laparoscopic instruments resecting tissue inside the patient's body cavity",
    "robotic instruments and fatty tissue visible inside the patient's abdominal cavity",
    "da vinci robotic instruments operating inside the patient's abdominal cavity",
]

OB_PROMPTS = [
    "doctor or surgeon face visible outside the patient's body in camera",
    "patient skin or body exterior is visible outside the surgical incision site",
    "a doctor's face is visible outside the patient's body while cleaning the camera lens",
    "doctor's face and operating room environment visible outside the surgical site",
    "surgical drapes and patient skin exterior visible outside the surgical site",
    "a doctor's or nurse's face is visible in the operating room camera",
    "doctor's face visible during trocar insertion outside the surgical site",
    "a doctor's or surgeon's face is visible outside the patient's body",
]


def compute_multi_text_embeddings(model, device):
    """Encode all prompts → (N_ib, 768) and (N_ob, 768), L2-normalised."""
    ib_feats, ob_feats = [], []
    with torch.no_grad():
        for text in IB_PROMPTS:
            tokens = surgvlp.tokenize(text, device=device)
            feat = model(inputs_text=tokens, mode="text")["text_emb"]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            ib_feats.append(feat.squeeze(0))
        for text in OB_PROMPTS:
            tokens = surgvlp.tokenize(text, device=device)
            feat = model(inputs_text=tokens, mode="text")["text_emb"]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            ob_feats.append(feat.squeeze(0))
    ib = torch.stack(ib_feats).to(device)
    ob = torch.stack(ob_feats).to(device)
    print(f"  IB prompts: {ib.shape}  OB prompts: {ob.shape}")
    return ib, ob


class GatedTextFusionClassifier(nn.Module):
    def __init__(self, backbone, n_ib=13, n_ob=8):
        super().__init__()
        self.backbone = backbone
        dim = 768
        self.text_proj = nn.Linear(dim, dim)
        self.ib_gate   = nn.Linear(dim, n_ib)
        self.ob_gate   = nn.Linear(dim, n_ob)
        self.cls_head  = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

    def get_img_features(self, imgs):
        return self.backbone(inputs_img=imgs, mode="video")["img_emb"]

    def classify(self, img_feat, ib_prompts, ob_prompts):
        ib_proj  = self.text_proj(ib_prompts)
        ob_proj  = self.text_proj(ob_prompts)
        ib_w     = torch.softmax(self.ib_gate(img_feat), dim=-1)
        ob_w     = torch.softmax(self.ob_gate(img_feat), dim=-1)
        ib_fused = ib_w @ ib_proj
        ob_fused = ob_w @ ob_proj
        fused    = torch.cat([img_feat, ib_fused, ob_fused], dim=-1)
        return self.cls_head(fused), ib_fused, ob_fused

    def forward(self, imgs, ib_prompts, ob_prompts):
        feat = self.get_img_features(imgs)
        logits, ib_fused, ob_fused = self.classify(feat, ib_prompts, ob_prompts)
        return logits, feat, ib_fused, ob_fused

    def set_train_mode(self):
        self.train()
        self.backbone.eval()
        for name, module in self.backbone.named_modules():
            if "layer4" in name:
                module.train()


@torch.no_grad()
def print_alignment_metrics(model, sample_feats, sample_labels, ib_prompts, ob_prompts):
    feats_n  = F.normalize(sample_feats, dim=-1)
    ib_imgs  = feats_n[sample_labels == 0]
    ob_imgs  = feats_n[sample_labels == 1]
    ib_raw   = F.normalize(ib_prompts.mean(0), dim=-1)
    ob_raw   = F.normalize(ob_prompts.mean(0), dim=-1)
    ib_mean_feat = sample_feats[sample_labels == 0].mean(0, keepdim=True)
    ob_mean_feat = sample_feats[sample_labels == 1].mean(0, keepdim=True)
    _, ib_fused_ib, _ = model.classify(ib_mean_feat, ib_prompts, ob_prompts)
    _, ib_fused_ob, ob_fused_ob = model.classify(ob_mean_feat, ib_prompts, ob_prompts)
    ib_fused_ib = F.normalize(ib_fused_ib.squeeze(0), dim=-1)
    ob_fused_ob = F.normalize(ob_fused_ob.squeeze(0), dim=-1)

    def mc(a, b):
        if a.dim() == 1: a = a.unsqueeze(0)
        if b.dim() == 1: b = b.unsqueeze(0)
        return (a @ b.T).mean().item()

    print(f"  {'cos(mean_ib_prompts, mean_ob_prompts)':<42} {mc(ib_raw, ob_raw):+.4f}")
    print(f"  {'cos(ib_img,  ob_img)':<42} {mc(ib_imgs, ob_imgs):+.4f}  (want LOW)")
    print(f"  {'cos(ib_prompt, ib_img)':<42} {mc(ib_raw, ib_imgs):+.4f}  (want HIGH ↑)")
    print(f"  {'cos(ob_prompt, ob_img)':<42} {mc(ob_raw, ob_imgs):+.4f}  (want HIGH ↑)")
    print(f"  {'cos(ib_fused[ib_query], ib_img)':<42} {mc(ib_fused_ib, ib_imgs):+.4f}  (want HIGH ↑)")
    print(f"  {'cos(ob_fused[ob_query], ob_img)':<42} {mc(ob_fused_ob, ob_imgs):+.4f}  (want HIGH ↑)")


@torch.no_grad()
def evaluate(model, test_videos_df, preprocess, ib_prompts, ob_prompts,
             device, args, pred_dir=None):
    model.eval()
    rows_raw, rows_smooth, aurocs, aps = [], [], [], []
    focal_fn = FocalLoss(gamma=2.0)
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
        df = test_videos_df[vid]
        labels_np = df["label"].map({"in-body": 0, "out-of-body": 1}).values

        all_probs = []
        loader = DataLoader(FrameDataset(df, preprocess),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        for imgs, labels_b in loader:
            imgs     = imgs.to(device)
            labels_b = labels_b.to(device)
            feat     = model.get_img_features(imgs)
            logits, _, _ = model.classify(feat, ib_prompts, ob_prompts)
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

    model = GatedTextFusionClassifier(backbone, n_ib=len(IB_PROMPTS), n_ob=len(OB_PROMPTS)).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable  : {total:,} params")

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Warm-start from  : {args.init_checkpoint}  (epoch {ckpt['epoch']}, F1={ckpt['f1_smooth']:.4f})")

    print("\nLoading label CSVs …")
    train_csvs = [gt_dir / f"{v}_labels.csv" for v in TRAIN_VIDEOS]
    if args.pseudo_dir:
        train_csvs += sorted(Path(args.pseudo_dir).glob("*_labels.csv"))
        print(f"  Pseudo-label CSVs from {args.pseudo_dir}")

    train_ds = OoBDataset(train_csvs, preprocess, sample_every=args.sample_every)
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
    head_params   = (list(model.text_proj.parameters()) +
                     list(model.ib_gate.parameters()) +
                     list(model.ob_gate.parameters()) +
                     list(model.cls_head.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": layer4_params, "lr": args.lr_backbone},
        {"params": head_params,   "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = torch.load(args.init_checkpoint, map_location="cpu")["f1_smooth"] \
              if args.init_checkpoint else 0.0

    sample_ds = OoBDataset([gt_dir / f"{v}_labels.csv" for v in TRAIN_VIDEOS],
                           preprocess, sample_every=args.sample_every)

    def get_sample_feats(n=2000):
        idx    = torch.randperm(len(sample_ds))[:n].tolist()
        subset = torch.utils.data.Subset(sample_ds, idx)
        loader = DataLoader(subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
        feats, lbls = [], []
        with torch.no_grad():
            model.eval()
            for imgs_s, lbl_s in loader:
                f = model.get_img_features(imgs_s.to(device))
                feats.append(f.cpu()); lbls.append(lbl_s)
        return torch.cat(feats), torch.cat(lbls)

    print(f"\n{'='*70}\nEpoch 00 — BASELINE\n{'='*70}")
    sf, sl = get_sample_feats()
    print_alignment_metrics(model, sf.to(device), sl.to(device), ib_prompts, ob_prompts)
    evaluate(model, test_videos_df, preprocess, ib_prompts, ob_prompts, device, args)

    for epoch in range(1, args.epochs + 1):
        model.set_train_mode()
        total_focal = total_align = total_n = total_correct = 0

        for imgs, labels_b in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False):
            imgs     = imgs.to(device)
            labels_b = labels_b.to(device)
            B        = imgs.shape[0]

            logits, feat, ib_fused, ob_fused = model(imgs, ib_prompts, ob_prompts)
            loss_focal = focal(logits, labels_b)

            ib_mask = (labels_b == 0)
            ob_mask = (labels_b == 1)
            loss_align = torch.tensor(0.0, device=device)
            n_terms = 0
            if ib_mask.sum() > 0:
                ib_feats_n = F.normalize(feat[ib_mask].detach(), dim=-1)
                ib_fused_n = F.normalize(ib_fused[ib_mask], dim=-1)
                loss_align = loss_align + (1 - F.cosine_similarity(ib_fused_n, ib_feats_n)).mean()
                n_terms += 1
            if ob_mask.sum() > 0:
                ob_feats_n = F.normalize(feat[ob_mask].detach(), dim=-1)
                ob_fused_n = F.normalize(ob_fused[ob_mask], dim=-1)
                loss_align = loss_align + (1 - F.cosine_similarity(ob_fused_n, ob_feats_n)).mean()
                n_terms += 1
            if n_terms > 0:
                loss_align = loss_align / n_terms

            loss = loss_focal + args.lambda_align * loss_align
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            total_focal   += loss_focal.item() * B
            total_align   += loss_align.item() * B
            total_n       += B
            total_correct += (logits.argmax(1) == labels_b).sum().item()

        scheduler.step()
        train_acc = total_correct / total_n

        print(f"\n{'='*70}")
        print(f"Epoch {epoch:02d}/{args.epochs}  focal={total_focal/total_n:.4f}  "
              f"align={total_align/total_n:.4f}  train_acc={train_acc:.4f}")
        print(f"{'='*70}")

        avg_f1_raw, avg_f1_smooth, avg_auroc, avg_ap, avg_val_loss, avg_val_acc = evaluate(
            model, test_videos_df, preprocess, ib_prompts, ob_prompts, device, args)

        sf, sl = get_sample_feats()
        print_alignment_metrics(model, sf.to(device), sl.to(device), ib_prompts, ob_prompts)

        marker = " ← best" if avg_f1_smooth > best_f1 else ""
        print(f"\n  Epoch {epoch:02d}  focal={total_focal/total_n:.4f}  acc={train_acc:.4f}  "
              f"F1_raw={avg_f1_raw:.4f}  F1_smooth={avg_f1_smooth:.4f}{marker}")

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
    print(f"\n{'='*70}\nDone.  Best F1 (smooth) = {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lr-backbone",     type=float, default=1e-4)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--sample-every",    type=int,   default=1)
    parser.add_argument("--lambda-align",    type=float, default=0.5)
    parser.add_argument("--smooth-window",   type=int,   default=25)
    parser.add_argument("--min-oob-frames",  type=int,   default=10)
    parser.add_argument("--pseudo-dir",      type=str,   default=None)
    parser.add_argument("--init-checkpoint", type=str,   default=None)
    parser.add_argument("--ckpt-dir",        type=str,   default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--ckpt-name",       type=str,   default="best_gated.pt")
    args = parser.parse_args()
    main(args)
