# SurgAnon: Out-of-Body Detection in Robotic Surgical Video via Gated Multi-Prompt Text Fusion with Temporal Modeling

Official PyTorch implementation of *"SurgAnon: Out-of-Body Detection in Robotic Surgical Video via Gated Multi-Prompt Text Fusion with Temporal Modeling"*

[Suyash Kumar](mailto:suyash.kumar.cd.mec22@itbhu.ac.in)<sup>1,2 †</sup> · [Sebastian Frey](mailto:frey.s@chu-nice.fr)<sup>1,3 †</sup> · Ezem Sura Ekmekci<sup>1</sup> · Duccio Lalli<sup>1,5,6</sup> · Quentin Rudondy<sup>3</sup> · Pierre Berthet-Rayne<sup>4</sup> · Hervé Delingette<sup>1</sup> · François Bremond<sup>1</sup> · Nicholas Ayache<sup>1</sup>

<sup>1</sup>Université Côte d'Azur, Inria, Sophia-Antipolis, France · <sup>2</sup>IIT (BHU) Varanasi, India · <sup>3</sup>CHU Nice, France · <sup>4</sup>Carvolix, Nice, France · <sup>5</sup>EURECOM, Sophia Antipolis, France · <sup>6</sup>Politecnico di Torino, Italy

<sup>†</sup>Equal contribution

---

**GatedTextFusion-LSTM** for privacy-preserving surgical video anonymisation. Detects frames where the laparoscopic camera is outside the patient body, achieving F1_smooth = 0.9849 and AP = 0.9956 on 17 da Vinci robotic procedures across 12 procedure types.

---

## Overview

The pipeline has three stages:

1. **Frame extraction** — extract frames from raw videos at 5 fps
2. **Label generation** — convert timestamp annotations to per-frame CSVs
3. **Model training & evaluation** — four model variants with optional semi-supervised extension

```
Raw videos
    │
    ▼ src/frame_extractor.py
Frames at 5 fps
    │
    ▼ timestamp annotations → label CSVs (one per video)
    │
    ├─ Stage 1: GatedTextFusion MLP  (models/train_gated.py)
    │
    └─ Stage 2: GatedTextFusion LSTM (models/train_gated_lstm.py)
                      │
                      ▼ eval/gen_pseudo_labels.py
               Pseudo-labeled videos
                      │
                      ▼ models/train_gated_lstm_semisup.py
               Semi-supervised model
```

---

## Installation

**System dependency (install separately):**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

**Python environment:**
```bash
conda create -n surganon python=3.10
conda activate surganon

# Install surgvlp from source
pip install git+https://github.com/camma-public/surgvlp

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Configuration

Edit `src/config.yaml` before running anything:

```yaml
data:
  frames_dir: "/path/to/frames_5fps"   # where extracted frames will be stored
  fps: 5
  train_videos:                         # video names used for training
    - video_001
    - video_002
    - video_003
    - video_004
    - video_005
  test_videos:                          # video names used for evaluation
    - video_006
    - video_007
    # ...

step1:
  output_dir: "/path/to/labels"        # where label CSVs will be stored
```

Each video name in `train_videos`/`test_videos` must correspond to:
- A subdirectory under `frames_dir`: `frames_dir/{video_name}/`
- A label CSV at: `step1.output_dir/{video_name}_labels.csv`

---

## Step 1: Frame Extraction

```python
from src.frame_extractor import extract_frames

extract_frames(
    video_path="/path/to/video_001.mp4",
    output_dir="/path/to/frames_5fps/video_001",
    fps=5
)
```

---

## Step 2: Label CSVs

Label CSVs are generated from your timestamp annotations (OoB segment start/end times). Each CSV must have the columns:

| Column | Description |
|--------|-------------|
| `frame_idx` | Frame index (0-based) |
| `timestamp_sec` | Timestamp in seconds |
| `frame_path` | Absolute path to the frame image |
| `label` | `in-body` or `out-of-body` |

One CSV per video, named `{video_name}_labels.csv`, placed in `step1.output_dir`.

---

## Step 3: Training

All training scripts read video names from `src/config.yaml` and label CSVs from `step1.output_dir`. Checkpoints are saved to `--ckpt-dir`.

### Ablation A — Visual-Only MLP (no text, no LSTM)

```bash
python models/train_visual_only.py \
  --epochs 15 \
  --lr 1e-3 --lr-backbone 1e-4 \
  --batch-size 256 \
  --ckpt-dir checkpoints \
  --ckpt-name best_visual_only.pt
```

### Ablation B — Visual-Only LSTM (no text)

Warm-starts from the Visual-Only MLP checkpoint.

```bash
python models/train_visual_only_lstm.py \
  --init-checkpoint checkpoints/best_visual_only.pt \
  --epochs 15 \
  --lr 1e-3 \
  --seq-len 64 --stride 32 \
  --hidden 512 --num-layers 2 \
  --ckpt-dir checkpoints \
  --ckpt-name best_visual_only_lstm.pt
```

### Ablation C — GatedTextFusion MLP (text, no LSTM) — Stage 1

```bash
python models/train_gated.py \
  --epochs 15 \
  --lr 1e-3 --lr-backbone 1e-4 \
  --batch-size 256 \
  --lambda-align 0.5 \
  --ckpt-dir checkpoints \
  --ckpt-name best_gated.pt
```

### Full Model — GatedTextFusion LSTM — Stage 2 (recommended)

Warm-starts LSTM head from the GatedTextFusion MLP checkpoint. Backbone is re-frozen for stage 2.

```bash
python models/train_gated_lstm.py \
  --init-checkpoint checkpoints/best_gated.pt \
  --epochs 15 \
  --lr 1e-3 \
  --seq-len 64 --stride 32 \
  --hidden 512 --num-layers 2 \
  --unfreeze-epoch 4 \
  --grad-accum 4 \
  --lambda-align 0.5 \
  --ckpt-dir checkpoints \
  --ckpt-name best_gated_lstm.pt
```

---

## Step 4: Evaluation

Each model variant has a corresponding eval script. All eval scripts require `--checkpoint` and `--pred-dir`.

```bash
# GatedTextFusion LSTM (full model)
python eval/eval_gated_lstm.py \
  --checkpoint checkpoints/best_gated_lstm.pt \
  --pred-dir predictions/gated_lstm \
  --seq-len 64 --hidden 512 --num-layers 2

# GatedTextFusion MLP
python eval/eval_gated_mlp.py \
  --checkpoint checkpoints/best_gated.pt \
  --pred-dir predictions/gated_mlp

# Visual-Only LSTM
python eval/eval_visual_only_lstm.py \
  --checkpoint checkpoints/best_visual_only_lstm.pt \
  --pred-dir predictions/visual_only_lstm \
  --seq-len 64 --hidden 512 --num-layers 2

# Visual-Only MLP
python eval/eval_visual_only_mlp.py \
  --checkpoint checkpoints/best_visual_only.pt \
  --pred-dir predictions/visual_only_mlp
```

**Output per video:** `{pred-dir}/{video}_pred.csv` with columns `frame_idx`, `timestamp_sec`, `frame_path`, `label`, `gt_label`, `confidence`, `correct`.

**Metrics reported:** F1_raw, F1_smooth, AUROC, AP, per-video TP/TN/FP/FN.

### Smoothing baseline predictions (OoBNet / IODA)

The paper applies identical post-processing to all models. If you have raw prediction CSVs from OoBNet or IODA (same `{video}_pred.csv` format with `label` and `confidence` columns), apply the same majority-vote + short-segment suppression:

```bash
python eval/smooth_baseline_preds.py \
  --oobnet-src /path/to/oobnet_raw_preds \
  --oobnet-dst /path/to/oobnet_smoothed_preds \
  --ioda-src   /path/to/ioda_raw_preds \
  --ioda-dst   /path/to/ioda_smoothed_preds \
  --smooth-window 25 --min-oob-frames 10 --threshold 0.5
```

This produces smoothed CSVs in `--oobnet-dst` / `--ioda-dst` with the same schema as the input, ready for metric computation or the demo viewer.

---

## Step 5: Semi-Supervised Extension (optional)

### 5a — Generate pseudo-labels for unlabeled videos

Unlabeled videos need frame-extracted subdirectories and placeholder CSVs (with `frame_path` column) in `--pseudo-input-dir`. The script re-labels them using the trained model.

```bash
python eval/gen_pseudo_labels.py \
  --checkpoint checkpoints/best_gated_lstm.pt \
  --pseudo-input-dir /path/to/unlabeled_labels \
  --pseudo-output-dir /path/to/pseudo_labels \
  --top-n 15 \
  --smooth-window 25 \
  --min-oob-frames 10
```

Outputs:
- `{pseudo-output-dir}/{video}_labels.csv` — re-labeled CSV per video
- `{pseudo-output-dir}/confidence_ranking.csv` — videos ranked by average confidence

### 5b — Semi-supervised fine-tuning

Fine-tunes only the input projection, LSTM, and classifier (backbone + gates frozen).

```bash
python models/train_gated_lstm_semisup.py \
  --init-checkpoint checkpoints/best_gated_lstm.pt \
  --pseudo-dir /path/to/pseudo_labels \
  --epochs 3 \
  --lr 2e-4 \
  --lambda-align 0.2 \
  --ckpt-dir checkpoints \
  --ckpt-name best_gated_lstm_semisup.pt
```

---

## Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 15 | Training epochs |
| `--lr` | 1e-3 | Head / task learning rate |
| `--lr-backbone` | 1e-4 | Backbone layer4 learning rate (MLP scripts only) |
| `--batch-size` | 256 | Batch size (MLP models) |
| `--seq-len` | 64 | LSTM window length in frames |
| `--stride` | 32 | LSTM window stride (overlapping windows) |
| `--hidden` | 512 | LSTM hidden dimension |
| `--num-layers` | 2 | LSTM depth |
| `--smooth-window` | 25 | Temporal majority-vote window for post-processing |
| `--min-oob-frames` | 10 | Minimum OoB segment length (shorter segments suppressed) |
| `--lambda-align` | 0.5 | Alignment loss weight (gated models only) |
| `--grad-accum` | 4 | Gradient accumulation steps (LSTM training) |
| `--unfreeze-epoch` | 4 | Epoch at which backbone layer4 is unfrozen (LSTM) |

---

## Repository Structure

```
SurgAnon/
├── src/
│   ├── config.yaml              # Data paths and video splits
│   ├── frame_extractor.py       # FFmpeg frame extraction
│   └── vllm_labeler.py          # Zero-shot SurgVLP labeler
├── models/
│   ├── utils.py                 # Dataset, FocalLoss, post-processing
│   ├── train_visual_only.py     # Ablation: visual MLP
│   ├── train_visual_only_lstm.py  # Ablation: visual LSTM
│   ├── train_gated.py           # Stage 1: GatedTextFusion MLP
│   ├── train_gated_lstm.py      # Stage 2: GatedTextFusion LSTM (full model)
│   └── train_gated_lstm_semisup.py  # Semi-supervised fine-tuning
├── eval/
│   ├── gen_pseudo_labels.py     # Generate pseudo-labels for unlabeled videos
│   ├── smooth_baseline_preds.py # Apply identical post-processing to OoBNet/IODA preds
│   ├── eval_visual_only_mlp.py
│   ├── eval_visual_only_lstm.py
│   ├── eval_gated_mlp.py
│   └── eval_gated_lstm.py
└── requirements.txt
```

---

## Results

All models trained on 5 videos (~7h 32min), evaluated on 12 held-out robotic procedures (~16h 54min) across 12 procedure types.

| Model | F1_smooth | AP | FP | FN |
|-------|-----------|-----|-----|-----|
| OoBNet (supervised) | 0.9390 | 0.9572 | 43 | 1 |
| IODA (supervised) | 0.9663 | 0.9883 | 8 | 6 |
| Visual-Only MLP | 0.9796 | 0.9902 | 41 | 2 |
| Visual-Only LSTM | 0.9808 | 0.9868 | 35 | 2 |
| GatedTextFusion MLP | 0.9791 | 0.9923 | 46 | 1 |
| **GatedTextFusion LSTM** | **0.9849** | **0.9956** | **19** | **1** |
| + semi-supervised | 0.9840 | **0.9964** | 18 | 1 |

FP/FN = total spurious/missed OoB segments across 12 test videos (post-smoothing).

---

## Pretrained Weights

The best-performing **GatedTextFusion LSTM** checkpoint (F1_smooth = 0.9849, AP = 0.9956) is available on HuggingFace:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="suyashkumar234/SurgAnon",
    filename="best_gated_lstm_weights.pt"
)
```

Then load into the model:

```python
import torch
model.load_state_dict(torch.load(path, map_location="cpu"))
```

Or download directly via CLI:

```bash
huggingface-cli download suyashkumar234/SurgAnon best_gated_lstm_weights.pt --local-dir checkpoints
```

---

## License

Code released for research use. The surgical video dataset cannot be released due to patient privacy, consistent with field norms followed by OoBNet and IODA.
