# AGENTS.md

## Environment

- Python **3.13** (see `.python-version`)
- use `uv` as my main python package manager
- Project environment lives at `.venv/` and is managed with `uv sync`
- Project dependencies are declared in `pyproject.toml` and locked in `uv.lock`

## Project structure

- `main.py` — stub entrypoint (no training/inference pipeline wired yet)
- `pyproject.toml` / `uv.lock` — uv-managed dependencies and locked environment
- `PROJECT_PLAN.md` — implementation plan for the context-aware sign language translation system
- `README.md` — usage notes for dataset layout and feature extraction
- `src/` — project source code
  - `src/dataset/preprocessing.py` — frame loading, sliding-window generation, and VideoMAE preprocessing
- `scripts/` — runnable utilities
  - `scripts/extract_features.py` — VideoMAE feature extraction to LMDB + metadata CSV generation
  - `scripts/test_single_video.py` — one-off smoke test for feature extraction on a single video group
- `tests/` — regression coverage
  - `tests/test_extract_features.py` — tests for empty videos, transient extraction errors, and LMDB key migration
- `datasets/` — How2Sign dataset inputs expected by the extractor
  - `datasets/how2sign_realigned_train.csv`
  - `datasets/how2sign_realigned_test.csv`
  - `datasets/val_rgb_front_clips/how2sign_realigned_val.csv`
  - `datasets/*_rgb_front_clips/raw_videos/` — sentence-level clip videos named `<SENTENCE_NAME>.mp4`
- `features/` — extracted outputs
  - `features/<split>_features.lmdb` — LMDB entries `{SENTENCE_NAME}/{feat_idx:07d}.np` plus `{SENTENCE_NAME}/done`
  - `features/<split>_metadata.csv` — metadata with sentence text and previous-sentence context
- `models/` — local LLM checkpoints (currently `Meta-Llama-3-8B/`)
- `LiTFiC/` — upstream reference implementation used as architecture reference
- `papers/` — reference papers, including the sign language translation paper for this project

## Commands

- `uv sync` — install/update the project environment
- `uv run python main.py` — runs the single stub entrypoint inside the project env
- `uv run pytest` — run tests inside the project env

## Status

- Tests exist for feature extraction regression coverage; no lint config, formatter, or CI yet
- No git commits yet (repo initialized, branch `master`)
