# Project Plan: Context-Aware Sign Language Translation for How2Sign

**Based on:** LiTFiC (Jang et al., arXiv 2025)  
**Dataset:** How2Sign (American Sign Language)  
**Hardware:** RTX 2060 (6GB VRAM)

---

## 1. Project Overview

### Objective
Build a sign language translation system that translates How2Sign (ASL) videos into English text using contextual cues.

### Architecture Summary
```
┌─────────────────────────────────────────────────────┐
│ Input Components (3)                                │
│ 1. Previous Sentence (auto-regressive context)      │
│ 2. Pseudo-gloss (CSLR model predictions)            │
│ 3. Visual Features (Frozen VideoMAE)                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ 3-Layer MLP Mapping Network + Temporal Convs        │
│ (768 → 4096 projection for How2Sign)                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ LLM Decoder (Llama3-8B or Qwen2.5-7B)               │
│ LoRA Fine-tuning (rank=4, alpha=16, dropout=0.05)   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ English Translation Output                          │
└─────────────────────────────────────────────────────┘
```

---

## 2. Key Differences from LiTFiC

| Aspect | LiTFiC | Our Approach |
|--------|--------|--------------|
| **Dataset** | BOBSL (BSL, TV broadcast) | How2Sign (ASL, instructional) |
| **Visual Encoder** | Video-Swin (BOBSL ISLR-pretrained) | **Frozen** VideoMAE (Kinetics-pretrained substitute) |
| **Background Cues** | BLIP2 captions (4th component) | **Not used** (3 components only) |
| **Mapping Network** | 2-layer MLP (BOBSL) | **3-layer MLP** (How2Sign, per LiTFiC) |
| **LLM** | Llama3-8B only | Llama3-8B **or** Qwen2.5-7B (experiment) |

---

## 3. Component Specifications

### 3.1 Visual Feature Extractor (Frozen)

| Specification | Value |
|---------------|-------|
| **Model** | VideoMAE Base (substitute for VideoSwin — no PyTorch HF model available) |
| **Pretraining** | Kinetics-400/700 |
| **Input** | Sliding window of 16 frames, 224×224, RGB |
| **Output** | 768-dim feature vector **per window/clip** (mean pool over patch tokens) |
| **Training** | **FROZEN** (feature extraction only) |
| **Hugging Face** | `MCG-NJU/videomae-base-finetuned-kinetics` |

#### Sliding Window Extraction (from LiTFiC)

```
For each sentence video:
  1. Load all frames (decord, 24fps)
  2. Resize each frame to 224×224, normalize (ImageNet mean/std)
  3. Sliding window: window_size=16 frames, stride=2 frames
  4. Each window → VideoMAE (frozen) → 768-dim vector
  5. Collect all T windows → shape [T, 768]

Example: 6.9s sentence @ 24fps = ~165 frames
  T = (165 - 16) / 2 + 1 ≈ 75 feature vectors
```

| Parameter | Value | Source |
|-----------|-------|--------|
| **Window size** | 16 frames | LiTFiC (same for BOBSL & How2Sign) |
| **Stride** | 2 frames | LiTFiC (stride s=2) |
| **Frame size** | 224×224 | VideoMAE input size |
| **Normalization** | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | ImageNet |
| **Output shape** | [T, 768] — variable T per sentence | LiTFiC |
| **Avg T (val set)** | ~75 windows (6.9s avg @ 24fps) | Estimated |

### 3.2 Pseudo-Gloss Generator (TO BE DETERMINED)

| Specification | Value |
|---------------|-------|
| **Model** | CSLR model pretrained on How2Sign |
| **Status** | **Need to find or train** |
| **Options** | 1. Find existing pretrained model<br>2. Train own (VideoMAE or another CSLR backbone + CTC)<br>3. Start without, add later |
| **Vocabulary** | ~1,000-3,000 glosses (estimated for How2Sign) |

### 3.3 Previous Sentence Context

| Specification | Value |
|---------------|-------|
| **Training** | 50% ground truth, 50% model prediction |
| **Inference** | Auto-regressive (model's own predictions) |
| **Purpose** | Resolve pronouns, tense, topic continuity |

### 3.4 Mapping Network (How2Sign Variant)

| Specification | Value |
|---------------|-------|
| **Architecture** | 3-layer MLP with temporal convolutions |
| **Input dim** | 768 (VideoMAE output) |
| **Output dim** | 4096 (LLM embedding space) |
| **Activation** | GELU |
| **Temporal** | 1D convolutions for frame sequence modeling |

### 3.5 LLM Decoder

| Specification | Llama3-8B | Qwen2.5-7B |
|---------------|-----------|------------|
| **Parameters** | 8B | 7B |
| **Context Length** | 8K tokens | 32K tokens |
| **Multilingual** | Good | Better |
| **SLT Precedent** | Used in LiTFiC | Less tested |
| **LoRA Config** | rank=4, alpha=16, dropout=0.05 | Same |

---

## 4. Dataset: How2Sign

### Characteristics
| Property | Value |
|----------|-------|
| **Sign Language** | American Sign Language (ASL) |
| **Domain** | Instructional videos (cooking, DIY, tutorials) |
| **Size** | ~80 hours, ~35K sentences |
| **Splits** | Train: 31,128 / Val: 1,741 / Test: 2,322 |
| **Vocabulary** | ~15.7K / 3.2K / 3.7K English words |
| **Frame Rate** | 24 fps |
| **Resolution** | 1280×720 (actual, will resize to 224×224) |

### Preprocessing Requirements
1. **Frame extraction** - Load frames via decord at native 24fps
2. **Resizing** - 224×224 for VideoMAE input
3. **Normalization** - mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
4. **Segmentation** - Videos are already sentence-segmented (one .mp4 per sentence)
5. **Sliding window** - window=16 frames, stride=2 → shape [T, 768] per sentence

---

## 5. Training Configuration

### Hardware Constraints
| Resource | Available | LiTFiC Reference |
|----------|-----------|------------------|
| **GPU** | 1x RTX 2060 (6GB) | 4x H100 (80GB each) |
| **VRAM** | 6GB | 320GB total |
| **Implication** | Smaller batch size, longer training | Batch size 2 per GPU |

### Proposed Training Settings
| Hyperparameter | Value |
|----------------|-------|
| **Batch Size** | 2-4 (adjust based on VRAM) |
| **Gradient Accumulation** | 4-8 steps |
| **Effective Batch** | 8-32 |
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Precision** | bfloat16 (if supported) or fp16 |
| **FlashAttention** | Yes (if available) |
| **Epochs** | 10-15 |
| **LoRA** | rank=4, alpha=16, dropout=0.05 |
| **LoRA Targets** | query_proj, value_proj only |

### Training Augmentations (from LiTFiC)
- **Word dropping:** Randomly omit 0-50% of words in textual cues
- **Cue dropping:** Randomly omit entire cues with 50% probability
- **Purpose:** Enable robust inference when cues are missing/noisy

---

## 6. Evaluation Metrics

| Metric | Purpose | Tool |
|--------|---------|------|
| **BLEU-4** | N-gram overlap | SacreBLEU |
| **BLEURT** | Semantic quality | BLEURT-20 |
| **ROUGE-L** | Recall-oriented overlap | rouge_score |
| **CIDEr** | Consensus-based evaluation | pycocoevalcap |
| **LLM Score** | Semantic adequacy (0-5 scale) | GPT-4o-mini or local LLM |

### LLM Evaluation Prompt (from LiTFiC)
```
Rate the translation quality from 0-5:
- 0: Completely wrong or unrelated
- 1: Mostly wrong with some relevant words
- 2: Partially correct but significant errors
- 3: Mostly correct with minor errors
- 4: Nearly perfect with very minor issues
- 5: Perfect translation

Reference: {reference}
Prediction: {prediction}
```

---

## 7. Implementation Phases

### Phase 1: Setup & Preprocessing (Week 1-2)
- [x] Set up project structure and install dependencies
- [x] Copy CSV to project, link videos in `datasets/val_rgb_front_clips/raw_videos/`
- [x] Implement `scripts/extract_features.py`:
  - Load each sentence video with decord
  - Apply sliding window (window=16, stride=2) over all frames
  - Pass each window through frozen VideoMAE → 768-dim
  - Store [T, 768] feature tensor per sentence
- [x] Build `features/val_features.lmdb` (LMDB entries keyed by `{SENTENCE_NAME}/{feat_idx:07d}.np`)
- [x] Include metadata CSV with sentence text, previous sentence, video name, and timestamps
- [ ] Confirm train split extraction is complete (`features/train_features.lmdb` + `features/train_metadata.csv`)

### Phase 2: Baseline Model (Week 2-3)
- [x] Implement 3-layer MLP mapping network (VisualProjector in `src/models/projector.py`)
- [x] Implement Llama3-8B + LoRA integration (VisualLLMBaseline in `src/models/visual_llm_baseline.py`)
- [x] Auto-infer LLM hidden size from pretrained config (no manual embedding_size argument)
- [x] Create `src/data/how2sign_lmdb_dataset.py` (LMDB feature dataset + variable-length collate)
- [x] Implement BLEU and ROUGE-L metrics in `src/training/metrics.py`
- [x] Create training utilities (seeding, optimizer, checkpointing) in `src/training/train_utils.py`
- [x] Implement `scripts/train_baseline.py` (train + validate each epoch)
- [x] Implement `scripts/eval_baseline.py` (load checkpoint, generate, score)
- [x] Visual-only baseline (no previous-sentence context, no pseudo-gloss)
- [ ] Run training on train split and evaluate baseline metrics

### Phase 3: Context Integration (Week 3-4)
- [ ] Add previous sentence context
- [ ] Implement auto-regressive training (50% GT, 50% pred)
- [ ] Evaluate improvement from context

### Phase 4: Pseudo-Gloss Integration (Week 4-6)
- [ ] **Option A:** Find pretrained CSLR model for How2Sign
- [ ] **Option B:** Train own CSLR model (e.g. VideoMAE + CTC)
- [ ] Integrate pseudo-gloss predictions
- [ ] Evaluate full model (all 3 components)

### Phase 5: LLM Comparison (Week 6-7)
- [ ] Train Qwen2.5-7B variant
- [ ] Compare Llama3 vs Qwen2.5 performance
- [ ] Select best model for final evaluation

### Phase 6: Final Evaluation & Analysis (Week 7-8)
- [ ] Run full evaluation on test set
- [ ] Ablation studies (each component's contribution)
- [ ] Qualitative analysis (error cases, success cases)
- [ ] Write report/thesis

---

## 8. Current Project Structure

```
ACV_final_project_new/
├── PROJECT_PLAN.md               # This file
├── README.md                     # Setup, dataset layout, feature extraction usage
├── AGENTS.md                     # Project-specific guidance for agent sessions
├── main.py                       # Stub entry point
├── pyproject.toml                # uv-managed dependencies
├── uv.lock                       # Locked dependency set
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── how2sign_lmdb_dataset.py   # LMDB feature dataset + variable-length collate
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── preprocessing.py           # Frame loading, sliding windows, VideoMAE preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── projector.py              # 3-layer MLP mapping network (768 → llm_hidden_size)
│   │   └── visual_llm_baseline.py    # Projector + LoRA LLM decoder wrapper
│   └── training/
│       ├── __init__.py
│       ├── metrics.py                # SacreBLEU and ROUGE-L computation
│       └── train_utils.py            # Seeding, optimizer, checkpointing, mixed precision
├── scripts/
│   ├── eval_baseline.py              # Load checkpoint and evaluate on a data split
│   ├── extract_features.py           # Frozen VideoMAE feature extraction to LMDB + metadata CSV
│   ├── train_baseline.py             # Phase 2 baseline training (visual-only, LoRA)
│   └── test_single_video.py          # One-off smoke test for extraction on a single video group
├── tests/
│   ├── test_extract_features.py      # Regression tests for extraction edge cases and LMDB migration
│   ├── test_how2sign_lmdb_dataset.py # Tests for feature dataset loader
│   ├── test_metrics.py               # Tests for BLEU/ROUGE-L metrics
│   ├── test_projector.py             # Tests for 3-layer mapping network
│   └── test_visual_llm_baseline.py   # Tests for projector+decoder baseline module
├── datasets/
│   ├── how2sign_realigned_train.csv
│   ├── how2sign_realigned_test.csv
│   ├── val_rgb_front_clips/
│   │   ├── how2sign_realigned_val.csv
│   │   └── raw_videos/
│   ├── train_rgb_front_clips/
│   │   └── raw_videos/
│   └── test_rgb_front_clips/
│       └── raw_videos/
├── features/
│   ├── val_features.lmdb         # LMDB entries `{SENTENCE_NAME}/{feat_idx:07d}.np`
│   ├── val_metadata.csv          # Sentence text + previous sentence + timestamps
│   └── ...                       # Future train/test feature and metadata outputs
├── models/
│   └── Meta-Llama-3-8B/          # Local LLM checkpoint storage
├── LiTFiC/                       # Upstream reference implementation
└── papers/                       # Reference papers
```

---

## 9. Dependencies (Estimated)

```python
# Core
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.40.0  # Llama, Qwen, VideoMAE
peft>=0.10.0  # LoRA
accelerate>=0.25.0

# Dataset
decord  # Video loading
opencv-python  # Frame extraction
pillow  # Image processing

# Evaluation
sacrebleu  # BLEU
bert-score  # BLEURT
rouge_score  # ROUGE
pycocoevalcap  # CIDEr

# Utilities
hydra-core  # Configuration
wandb  # Experiment tracking
tqdm  # Progress bars
```

---

## 10. Pending Decisions

| Decision | Options | Status |
|----------|---------|--------|
| **Pseudo-gloss model** | Find pretrained / Train own / Skip initially | **TODO** |
| **LLM choice** | Llama3-8B / Qwen2.5-7B / Both | **Experiment** |
| **Batch size** | 2 / 4 (based on VRAM testing) | **TODO** |
| **How2Sign access** | Already downloaded / Need to download | **TODO** |
| **LLM evaluation** | GPT-4o-mini / Local LLM (Qwen) | **TODO** |

---

## 11. Reference Papers

### Primary
1. **LiTFiC** - "Lost in Translation, Found in Context" (Jang et al., arXiv 2025)
   - Main architecture reference
   - Contextual cue integration
   - BOBSL benchmark

### Secondary
2. **SONAR-SLT** - "Multilingual Sign Language Translation via Language-Agnostic Sentence Embedding Supervision" (Hamidullah et al., arXiv 2025)
   - Alternative: language-agnostic semantic embeddings
   - 200+ language support

3. **Uni-Sign** - "Toward Unified Sign Language Understanding at Scale" (Li et al., ICLR 2025)
   - Unified SLT framework
   - CSL-News dataset (1,985 hours)

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **VRAM overflow** | Use gradient checkpointing, reduce batch size, gradient accumulation |
| **No pseudo-gloss model** | Start with 2-component baseline, add later |
| **LLM too large for 16GB** | Use QLoRA (4-bit), or smaller model (Phi-3, Mistral-7B) |
| **How2Sign preprocessing complex** | Use existing dataloaders (mmengine, decord) |
| **Training too slow** | Fewer epochs, early stopping, smaller validation set |

---

**Last Updated:** May 14, 2026  
**Status:** Phase 1 mostly complete (validation extraction done, train extraction pending); Phase 2-6 not started
