# ACV Final Project

## Setup

```bash
uv sync
```

## Download LLM

All Llama models are gated on HuggingFace. Request access first, then download with the scripts below.

### Llama-3-8B

```bash
HF_TOKEN=hf_xxxx uv run python scripts/download_llama.py
```

### Llama-3.2-1B

```bash
HF_TOKEN=hf_xxxx uv run python scripts/download_llama32_1b.py
```

### Llama-3.2-3B

```bash
HF_TOKEN=hf_xxxx uv run python scripts/download_llama32_3b.py
```

Override output directory:

```bash
uv run python scripts/download_llama.py --output-dir /path/to/save
```

## Dataset Paths

Place the dataset under `datasets/` using the paths expected by `scripts/extract_features.py`.

Expected layout:

```text
datasets/
├── how2sign_realigned_train.csv
├── how2sign_realigned_test.csv
├── val_rgb_front_clips/
│   ├── how2sign_realigned_val.csv
│   └── raw_videos/
├── train_rgb_front_clips/
│   └── raw_videos/
└── test_rgb_front_clips/
    └── raw_videos/
```

Default paths used by each split:

- `val`
  - CSV: `datasets/val_rgb_front_clips/how2sign_realigned_val.csv`
  - Videos: `datasets/val_rgb_front_clips/raw_videos/`
- `train`
  - CSV: `datasets/how2sign_realigned_train.csv`
  - Videos: `datasets/train_rgb_front_clips/raw_videos/`
- `test`
  - CSV: `datasets/how2sign_realigned_test.csv`
  - Videos: `datasets/test_rgb_front_clips/raw_videos/`

Each video file should be named as:

```text
<SENTENCE_NAME>.mp4
```

If your files are stored somewhere else, use path overrides:

```bash
uv run python scripts/extract_features.py \
  --split train \
  --csv /path/to/your_train.csv \
  --video-dir /path/to/train_videos/
```

## Extract Video Features

Run feature extraction with the uv-managed environment:

```bash
uv run python scripts/extract_features.py --split val
```

Available splits:
- `val`
- `train`
- `test`

Default outputs:
- LMDB features: `features/<split>_features.lmdb`
- Metadata CSV: `features/<split>_metadata.csv`

Example for the validation split:
- LMDB: `features/val_features.lmdb`
- Metadata: `features/val_metadata.csv`

Useful options:

```bash
uv run python scripts/extract_features.py \
  --split val \
  --device cuda \
  --batch-size 8
```

Write outputs to a custom folder:

```bash
mkdir -p outputs/features
uv run python scripts/extract_features.py \
  --split val \
  --output-lmdb outputs/features/val_features.lmdb \
  --output-metadata outputs/features/val_metadata.csv
```

Optional path overrides:
- `--csv <path>`
- `--video-dir <path>`
- `--output-lmdb <path>`
- `--output-metadata <path>`
- `--device <cpu|cuda>`
- `--batch-size <int>`
- `--map-size-gb <int>`

## Video-Only Training Recipe

Use this flow when you want to train the Phase 2 visual-only baseline end to end.

### Step 1: Extract train features

Default train extraction command:

```bash
uv run python scripts/extract_features.py \
  --split train \
  --device cuda \
  --batch-size 8
```

This writes:
- `features/train_features.lmdb`
- `features/train_metadata.csv`

If the machine is offline but already has the VideoMAE cache locally:

```bash
HF_HOME=~/.cache/huggingface uv run python scripts/extract_features.py \
  --split train \
  --device cuda \
  --batch-size 8
```

If you are not using `uv run`, make sure the repo root is on `PYTHONPATH`:

```bash
HF_HOME=~/.cache/huggingface PYTHONPATH=. python scripts/extract_features.py \
  --split train \
  --device cuda \
  --batch-size 8
```

The extractor auto-resumes and skips clips that already have `{SENTENCE_NAME}/done` in the LMDB.

### Step 2: Train the video-only baseline

After `train_features.lmdb` and `train_metadata.csv` exist, run:

```bash
uv run python scripts/train_baseline.py \
  --train-lmdb features/train_features.lmdb \
  --train-metadata features/train_metadata.csv \
  --val-lmdb features/val_features.lmdb \
  --val-metadata features/val_metadata.csv \
  --pretrained-llm models/Llama-3.2-1B \
  --output-dir outputs/video_only_baseline \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --epochs 5 \
  --device cuda
```

Recommended starting point on this machine:
- use `models/Llama-3.2-1B`
- keep `--batch-size 1`
- use gradient accumulation with `--grad-accum-steps 4`

**TensorBoard Logging:**

Training logs are automatically written to `<output-dir>/tensorboard`. To view:

```bash
tensorboard --logdir outputs/video_only_baseline/tensorboard
```

Then open the URL shown in your browser (usually `http://localhost:6006`).

### Suggested Commands

#### Llama-3.2-1B

```bash
uv run python scripts/train_baseline.py \
  --train-lmdb features/train_features.lmdb \
  --train-metadata features/train_metadata.csv \
  --val-lmdb features/val_features.lmdb \
  --val-metadata features/val_metadata.csv \
  --pretrained-llm models/Llama-3.2-1B \
  --output-dir outputs/video_only_llama32_1b \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --epochs 5 \
  --device cuda
```

#### Llama-3.2-3B

```bash
uv run python scripts/train_baseline.py \
  --train-lmdb features/train_features.lmdb \
  --train-metadata features/train_metadata.csv \
  --val-lmdb features/val_features.lmdb \
  --val-metadata features/val_metadata.csv \
  --pretrained-llm models/Llama-3.2-3B \
  --output-dir outputs/video_only_llama32_3b \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 5 \
  --device cuda
```

#### Meta-Llama-3-8B

```bash
uv run python scripts/train_baseline.py \
  --train-lmdb features/train_features.lmdb \
  --train-metadata features/train_metadata.csv \
  --val-lmdb features/val_features.lmdb \
  --val-metadata features/val_metadata.csv \
  --pretrained-llm models/Meta-Llama-3-8B \
  --output-dir outputs/video_only_llama3_8b \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 5 \
  --device cuda
```

#### Qwen Template

```bash
uv run python scripts/train_baseline.py \
  --train-lmdb features/train_features.lmdb \
  --train-metadata features/train_metadata.csv \
  --val-lmdb features/val_features.lmdb \
  --val-metadata features/val_metadata.csv \
  --pretrained-llm /path/to/Qwen-model \
  --output-dir outputs/video_only_qwen \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 5 \
  --device cuda
```

### Step 3: Evaluate the trained checkpoint

```bash
uv run python scripts/eval_baseline.py \
  --lmdb features/val_features.lmdb \
  --metadata features/val_metadata.csv \
  --checkpoint outputs/video_only_baseline/best.pt \
  --pretrained-llm models/Llama-3.2-1B \
  --output-dir outputs/video_only_eval
```

## Phase 2 Reference

The commands above are the recommended end-to-end workflow.

Useful training options:
- `--pretrained-llm` — path to a local HF-compatible LLM
- `--lora-r` / `--lora-alpha` / `--lora-dropout` — LoRA configuration
- `--precision` — `fp16` or `fp32`
- `--max-new-tokens` — max tokens for validation generation
- `--val-samples` — limit validation set size for faster eval

Training outputs in `--output-dir`:
- `best.pt` — checkpoint with best validation BLEU
- `last.pt` — final epoch checkpoint
- `val_predictions_epoch{N}.jsonl` — validation predictions
- `metrics.json` — full training metrics history

Evaluation outputs in `--output-dir`:
- `predictions.jsonl` — all model predictions with references
- `metrics.json` — BLEU and ROUGE-L scores
