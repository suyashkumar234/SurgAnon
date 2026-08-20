"""Shared dataset classes, loss, and post-processing utilities."""

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "src" / "config.yaml"))
TRAIN_VIDEOS: list[str] = _cfg["data"]["train_videos"]
TEST_VIDEOS:  list[str] = _cfg["data"]["test_videos"]


class OoBDataset(Dataset):
    def __init__(self, csv_paths, preprocess, sample_every=1):
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]
        dfs = []
        for p in csv_paths:
            p = Path(p)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df = df[df["label"].isin(["in-body", "out-of-body"])].reset_index(drop=True)
            df = df.iloc[::sample_every].reset_index(drop=True)
            dfs.append(df)
        self.df         = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        self.preprocess = preprocess
        self.label_map  = {"in-body": 0, "out-of-body": 1}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.preprocess(Image.open(row["frame_path"]).convert("RGB"))
        return img, self.label_map[row["label"]]


class FrameDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df         = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.label_map  = {"in-body": 0, "out-of-body": 1}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.preprocess(Image.open(row["frame_path"]).convert("RGB"))
        return img, self.label_map[row["label"]]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def sliding_majority_vote(flags, window=25):
    n, half = len(flags), window // 2
    return [int(sum(flags[max(0, i - half):min(n, i + half + 1)]) >
                (min(n, i + half + 1) - max(0, i - half)) / 2)
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
                for k in range(i, j): flags[k] = 0
            i = j
        else:
            i += 1
    return flags
