# Coursework Development Plan: Multimodal Transformer-Based Detection of Sarcasm and Hate Speech

## 1) Project framing and scope (Week 1)
- Define the exact coursework goal: build and evaluate a **multitask multimodal model** for sarcasm + hate speech in memes.
- Lock down target tasks:
  - Task A: sarcasm (binary)
  - Task B: hate speech (binary)
- Decide whether to train jointly on one combined dataset or via staged multi-dataset training (recommended).
- Finalize success criteria for the report:
  - Primary: Macro-F1, AUC-ROC
  - Secondary: precision/recall per class, calibration, confusion matrices

## 2) Reproducible repository setup (Week 1)
- Create a clean project structure:
  - `src/` (models, data, training, eval)
  - `configs/` (YAML for model/training/data)
  - `scripts/` (train, evaluate, inference)
  - `notebooks/` (EDA + error analysis only)
  - `reports/` (figures/tables for coursework)
- Set up environment management (Conda/venv + pinned `requirements.txt`).
- Add experiment tracking from day one (Weights & Biases or MLflow).
- Fix random seeds and deterministic flags for reproducibility.

## 3) Data acquisition, licensing, and split strategy (Week 1–2)
- Acquire benchmark datasets referenced in the README:
  - Hateful Memes
  - HarMeme
  - MSD (multimodal sarcasm)
- Document licensing and allowed coursework use.
- Standardize schema into a unified format:
  - `image_path`, `text`, `sarcasm_label`, `hate_label`, `source_dataset`
- Build **stratified 70/15/15 splits** per dataset where possible.
- Add leakage checks:
  - Hash-based duplicate detection across train/val/test.
  - Near-duplicate text detection for reposted meme captions.

## 4) Preprocessing pipeline implementation (Week 2)
### Text pipeline
- Unicode normalization (NFKC).
- Hashtag segmentation (Viterbi or fallback word-segmentation).
- Emoji-to-text mapping using CLDR names.
- Tokenization using `vinai/bertweet-large` tokenizer.
- Preserve punctuation and sarcasm markers (`?!`, ellipses, scare quotes).

### Image pipeline
- Resize to 224×224 with letterboxing (maintain aspect ratio).
- Train-time augmentation only:
  - horizontal flip (p=0.5)
  - light color jitter (±10%)
- Avoid aggressive cropping/rotation to preserve meme text context.

### Labels and imbalance
- Implement soft labels/label smoothing for noisy annotations.
- Use Focal Loss (`gamma=2.0`) for class imbalance.

## 5) Model implementation milestones (Week 2–3)
- Implement `MultimodalDetector` exactly as in README design:
  - CLIP-ViT-B/16 vision encoder
  - BERTweet-Large text encoder
  - projection layers to shared 512-d space
  - symmetric cross-attention fusion
  - shared MLP + two task heads
- Add modularity so fusion can be swapped (for ablations):
  - concat fusion
  - cross-attention fusion (main)
- Add robust shape/assertion checks in forward pass.

## 6) Training curriculum (Week 3)
### Stage 1: alignment tuning (5 epochs)
- Freeze CLIP + BERTweet backbones.
- Train only projection + fusion layers.
- Include auxiliary alignment loss (InfoNCE) with weight ~0.1.

### Stage 2: full fine-tuning (15 epochs)
- Unfreeze all layers.
- Use discriminative LRs:
  - encoders: 1e-5
  - fusion/heads: 1e-4
- Add warmup (500 steps), gradient clipping (1.0), weight decay (1e-2).
- Apply uncertainty-based multitask weighting (learnable sigmas).

## 7) Baselines and ablations (Week 3–4)
- Train required baselines:
  1. Text-only (BERTweet + MLP)
  2. Image-only (CLIP-ViT + MLP)
  3. Late fusion concat baseline
- Run ablations on proposed model:
  - Remove cross-attention (replace with concat)
  - Remove alignment loss
  - Single-task vs multitask training
- Use same splits, seeds, and metric pipeline for fairness.

## 8) Evaluation and significance testing (Week 4)
- For each model, run **5 random seeds**.
- Report:
  - Macro-F1, AUC-ROC (primary)
  - class-wise precision/recall/F1
  - mean ± std over seeds
- Perform paired t-test between proposed model and strongest baseline.
- Include calibration and threshold analysis if time permits.

## 9) Error analysis and explainability (Week 4–5)
- Build a structured error taxonomy:
  - OCR-heavy failure
  - implicit cultural context
  - irony without explicit polarity cues
  - reclaimed slurs/ambiguous hate language
- Visualize attention maps or token/patch importance for qualitative insights.
- Add short case studies comparing correct vs incorrect predictions.

## 10) Coursework deliverables (Week 5)
- Final report sections:
  1. Problem statement and motivation
  2. Related work (multimodal hate/sarcasm detection)
  3. Method (architecture + losses + training strategy)
  4. Experiments (datasets, setup, baselines)
  5. Results + ablations + significance
  6. Error analysis + ethics and limitations
  7. Conclusion and future work
- Prepare reproducibility appendix:
  - hardware
  - runtime
  - hyperparameters
  - exact command lines
- Optional demo notebook/script for inference on custom memes.

---

## Recommended timeline snapshot (5 weeks)
- **Week 1:** setup + data acquisition + split protocol
- **Week 2:** preprocessing + baseline-ready dataloaders
- **Week 3:** full model implementation + staged training
- **Week 4:** baselines, ablations, multi-seed experiments
- **Week 5:** analysis, writing, polishing

## Minimum viable submission (if time-constrained)
If deadlines become tight, prioritize:
1. One primary dataset + one external validation dataset.
2. Proposed cross-attention model + concat baseline.
3. Macro-F1/AUC + confusion matrix + 1–2 ablations.
4. Clear error analysis and honest limitations.
