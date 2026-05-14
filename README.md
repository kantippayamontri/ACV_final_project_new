# ACV Final Project

## Setup

```bash
uv sync
```

## Download LLM

Llama-3-8B is gated on HuggingFace. Request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B, then:

```bash
HF_TOKEN=hf_xxxx uv run python scripts/download_llama.py
```

Or override the output directory:

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

## Train Phase 2 Baseline

Train the visual-only baseline (VideoMAE features → projector → LoRA LLM decoder).

LLM hidden size is automatically inferred from the pretrained model config.

```bash
uv run python scripts/train_baseline.py \
  --train-lmdb features/train_features.lmdb \
  --train-metadata features/train_metadata.csv \
  --val-lmdb features/val_features.lmdb \
  --val-metadata features/val_metadata.csv \
  --pretrained-llm models/Meta-Llama-3-8B \
  --output-dir outputs/phase2-baseline \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --epochs 5 \
  --device cuda
```

Key options:
- `--pretrained-llm` — path to a local HF-compatible LLM (default: `models/Meta-Llama-3-8B`)
- `--lora-r` / `--lora-alpha` / `--lora-dropout` — LoRA configuration
- `--precision` — `fp16` or `fp32`
- `--max-new-tokens` — max tokens for validation generation (default: 128)
- `--val-samples` — limit validation set size for faster eval

Outputs per epoch (in `--output-dir`):
- `best.pt` — checkpoint with best val BLEU
- `last.pt` — final epoch checkpoint
- `val_predictions_epoch{N}.jsonl` — validation predictions
- `metrics.json` — full training metrics history

## Evaluate Phase 2 Baseline

Load a saved checkpoint and evaluate on any data split.

```bash
uv run python scripts/eval_baseline.py \
  --lmdb features/val_features.lmdb \
  --metadata features/val_metadata.csv \
  --checkpoint outputs/phase2-baseline/best.pt \
  --pretrained-llm models/Meta-Llama-3-8B \
  --output-dir outputs/phase2-eval
```

Outputs:
- `predictions.jsonl` — all model predictions with references
- `metrics.json` — BLEU and ROUGE-L scores
