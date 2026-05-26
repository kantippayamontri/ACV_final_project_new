## Auto-Regressive Video-Previous Evaluation Design

### Goal

Add a dedicated evaluation script for `VideoPrevLLM` that runs inference sentence-by-sentence in video order, using the model's prediction from the previous clip as the previous-sentence context for the next clip.

### Scope

This design covers:

- a new `scripts/eval_vdo_prev.py` evaluation entrypoint
- auto-regressive previous-sentence evaluation within each video
- checkpoint loading compatible with saved LoRA arguments
- metrics and prediction output files matching the existing evaluation workflow

This design does not cover:

- changes to training behavior
- Lightning parity
- beam search or alternative decoding strategies
- hybrid modes that mix ground-truth and predicted previous text during evaluation

### Current State

- `scripts/eval_baseline.py` evaluates `VisualLLMBaseline` without previous-sentence context
- `scripts/generate_prev_predictions.py` already contains the sequential, video-ordered generation pattern needed for auto-regressive context propagation
- `src/models/visual_prev_llm.py` exposes `generate(features, mask, prev_texts, max_new_tokens=128, **gen_kwargs)`
- `features/*_metadata.csv` rows contain `VIDEO_NAME`, `SENTENCE_NAME`, `SENTENCE`, and ordering fields such as `START_REALIGNED`

### Problem

The repository can train and generate previous-sentence-conditioned outputs, but it does not yet have a dedicated evaluation script for the real inference-time setup where each sentence receives the model output from the immediately previous sentence in the same video.

Using `eval_baseline.py` is insufficient because it ignores previous-sentence context entirely. Using ground-truth previous sentences at evaluation time would not match the intended deployment behavior.

### Chosen Approach

Create a new script, `scripts/eval_vdo_prev.py`, that mirrors the checkpoint-loading and metric-writing behavior of `scripts/eval_baseline.py` while using the sequential auto-regressive loop pattern from `scripts/generate_prev_predictions.py`.

For each metadata row, the script will:

1. reset previous context to `""` when a new `VIDEO_NAME` starts
2. load features for the current `SENTENCE_NAME`
3. call `VideoPrevLLM.generate(..., [previous_prediction])`
4. save the current output as the prediction for this sentence
5. carry that output forward as the next sentence's previous context

This matches the inference behavior the user explicitly approved.

### Alternatives Considered

#### 1. Extend `scripts/eval_baseline.py` with a mode flag

- Pros: one fewer script
- Cons: mixes visual-only and previous-sentence-conditioned evaluation paths into one file, increasing branching and reducing clarity

#### 2. Evaluate using precomputed previous-sentence JSON

- Pros: can reuse outputs from `generate_prev_predictions.py`
- Cons: evaluates a cached auxiliary artifact rather than the true step-by-step inference path; less direct for final model evaluation

#### 3. Evaluate with ground-truth `PREV_SENTENCE`

- Pros: simplest to implement
- Cons: does not match intended inference behavior and overestimates robustness

The dedicated auto-regressive script is the best fit because it keeps evaluation behavior explicit and aligned with real usage.

### Design Details

#### A. New script: `scripts/eval_vdo_prev.py`

Purpose: evaluate a saved `VideoPrevLLM` checkpoint in video order, using the model's own previous prediction as context.

Inputs:

- `--lmdb`
- `--metadata`
- `--checkpoint`
- `--pretrained-llm`
- `--output-dir`
- `--max-new-tokens`
- `--device`

Optional simplification:

- omit `--batch-size` from the first version because inference is inherently sequential across sentences inside a video
- omit `--num-workers` unless a later implementation clearly benefits from it

Behavior:

- load metadata rows from CSV
- convert `START_REALIGNED` to float and sort by `(VIDEO_NAME, START_REALIGNED)` to make ordering explicit
- fail if no rows are available
- open LMDB once for the full run
- load `VideoPrevLLM` using LoRA arguments stored in checkpoint metadata, matching the existing checkpoint-loading pattern used elsewhere in the repo
- iterate rows sequentially with a progress bar
- reset `previous_prediction = ""` on each new `VIDEO_NAME`
- read features for the current sentence using the same LMDB key conventions as existing scripts
- if a sentence has no features, record an empty prediction for that sentence and reset `previous_prediction = ""` before continuing
- otherwise call `model.generate(features.unsqueeze(0), mask.unsqueeze(0), [previous_prediction], max_new_tokens=...)`
- append the generated text to the predictions list and the reference sentence to the references list
- update `previous_prediction` with the generated text
- compute BLEU and ROUGE-L at the end
- save `predictions.jsonl` and `metrics.json` into `output-dir`

#### B. Shared utility behavior

The implementation should reuse existing patterns rather than inventing new abstractions:

- checkpoint loading should follow `scripts/eval_baseline.py`
- sequential metadata iteration and per-video reset should follow `scripts/generate_prev_predictions.py`
- LMDB feature reading should follow the existing 0-based plus legacy-1-based compatibility behavior already present in dataset/evaluation code

No new general-purpose utility module is needed for the first version.

#### C. Output contract

Outputs should mirror existing evaluation scripts so downstream usage stays familiar:

- `predictions.jsonl`: each prediction with its reference
- `metrics.json`: aggregate BLEU/ROUGE-L metrics

The script should also print a short summary with BLEU and ROUGE-L and the destination output directory.

### Data Flow

`metadata rows ordered by video/time + LMDB features + VideoPrevLLM checkpoint -> eval_vdo_prev.py -> autoregressive predictions + references -> metrics.json / predictions.jsonl`

Within a single video, the state transition is:

`previous_prediction_t -> model.generate(sentence_t) -> prediction_t -> previous_prediction_(t+1)`

At video boundaries, the state resets to `""`.

### Error Handling

- raise `ValueError` if metadata is empty
- raise a clear checkpoint-loading error if the checkpoint is unreadable or incompatible
- create `output-dir` if it does not exist
- if a sentence has no features, emit an empty prediction and reset chaining for the following sentence in that video

The chosen missing-feature behavior is conservative and matches the current previous-sentence workflow: it avoids leaking stale context forward when an intermediate clip cannot be evaluated.

### Testing Strategy

Tests should be added before implementation code.

Required coverage:

1. evaluation resets previous text on new video boundaries
2. sentence N receives sentence N-1's generated text as previous context within the same video
3. missing-feature rows emit an empty prediction and reset chaining for the next row
4. checkpoint loading restores saved LoRA arguments
5. output directory is created automatically
6. predictions and metrics files are written

Tests should stay focused and mostly unit-level, using fake models and small synthetic metadata/LMDB fixtures.

### Rollout Plan

1. add failing tests for auto-regressive evaluation behavior
2. implement `scripts/eval_vdo_prev.py` minimally to satisfy the tests
3. run focused tests for the new script
4. run broader regression tests for adjacent evaluation/checkpoint-loading behavior

### Example Command

```bash
PYTHONPATH=. uv run python scripts/eval_vdo_prev.py \
  --lmdb features/val_features.lmdb \
  --metadata features/val_metadata.csv \
  --checkpoint outputs/video_prev_run/best.pt \
  --pretrained-llm models/Llama-3.2-1B \
  --output-dir outputs/video_prev_run/eval_autoreg
```

### Success Criteria

- user can evaluate a `VideoPrevLLM` checkpoint with true auto-regressive previous-sentence context
- previous context resets correctly at video boundaries
- missing-feature cases do not leak stale context into later sentences
- outputs match the repository's existing evaluation artifact style
- focused tests cover the key sequencing and checkpoint-loading behaviors
