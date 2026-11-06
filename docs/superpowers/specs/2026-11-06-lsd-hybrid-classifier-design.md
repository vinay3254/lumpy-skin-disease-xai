# Lumpy Skin Disease Hybrid CNN–Transformer Classifier — Design

Date: 2026-11-06
Context: graduation major project, submission 2026-11-07.

## 1. Goal

Classify cattle images for Lumpy Skin Disease (LSD) with a hybrid CNN + Transformer
network that fuses local (CNN) and global (Transformer) features, and explain each
prediction with Grad-CAM, LIME and SHAP.

Success criteria:
- Binary (Healthy vs LSD) model trained and evaluated on a held-out test split.
- Ablation table: CNN-only, ViT-only, hybrid-concat, hybrid-attention.
- Report-ready figures: confusion matrix, ROC curve, training curves, XAI grids.
- Gradio demo: upload image → prediction, confidence, Grad-CAM + LIME overlay.

## 2. Scope

In scope:
- Binary classification (main result).
- 3-class extension (Healthy / LSD / Other skin disease) as a stretch goal, only if a
  downloadable "other cattle skin disease" dataset is found and binary work is done.

Out of scope (future work in report):
- Severity grading (early / moderate / severe). No public dataset has these labels.
- 5-class scheme from the proposal slide.
- k-fold cross-validation (single stratified split due to time).

## 3. Data

- Sources: Kaggle LSD datasets listed in the proposal (Kaggle Lumpy Skin Images,
  Kaggle LSD Dataset, Cow Lumpy Disease Dataset), downloaded via Kaggle API. Mendeley /
  Zenodo sets are added only if they download without manual steps.
- Merge into `data/raw/<source>/<class>/`.
- Deduplicate across all sources with perceptual hash (pHash, Hamming distance ≤ 4),
  so augmented copies cannot leak between splits.
- Stratified split 70 / 15 / 15 (train / val / test), fixed seed 42. Split saved as
  `data/splits.csv` (path, label, split, source).

## 4. Preprocessing and augmentation

- Resize to 224 × 224.
- Median filter (kernel 3) for noise removal.
- Normalize with ImageNet mean/std (inputs scaled to 0–1 first).
- Train-only augmentation: random rotation (±20°), horizontal flip, random resized crop
  (scale 0.8–1.0, "zoom"), color jitter (brightness/contrast 0.2).

## 5. Model

`HybridNet(num_classes, fusion, mode)`:
- CNN branch: EfficientNet-B0 (timm, ImageNet pretrained). Output: feature map
  (B, 1280, 7, 7) → projected to 256-d tokens (B, 49, 256) and pooled vector (B, 256).
- Transformer branch: DeiT-Tiny patch16 224 (timm, pretrained). Output: CLS + patch
  tokens (B, 197, 192) → projected to 256-d.
- Fusion:
  - `concat`: concatenate pooled CNN vector and ViT CLS vector → (B, 512).
  - `attention`: multi-head cross-attention (4 heads), CNN tokens as queries, ViT tokens
    as keys/values; mean-pool output, concatenate with ViT CLS → (B, 512).
- Classifier: Linear(512, 256) → GELU → Dropout(0.3) → Linear(256, num_classes).
- `mode`: `hybrid` | `cnn` (CNN branch + head only) | `vit` (ViT branch + head only),
  for ablation.

## 6. Training

- Script: `python -m src.train --model hybrid --fusion attention --classes 2`.
- AdamW, lr 3e-4 (head) / 1e-4 (backbones), weight decay 0.05, cosine schedule,
  batch 32, mixed precision, max 15 epochs, early stopping on val F1 (patience 4).
- Class-weighted cross-entropy for imbalance.
- Best checkpoint and history saved to `outputs/runs/<run_name>/`.

## 7. Evaluation

- `python -m src.evaluate --run <run_name>` on test split.
- Metrics: accuracy, precision, recall, F1 (macro), ROC-AUC; saved as `metrics.json`.
- Figures: confusion matrix, ROC curve, loss/accuracy curves.
- `python -m src.ablation` collects all runs into `outputs/ablation.csv` and `.md`.

## 8. Explainability

- Grad-CAM on last CNN conv block of the hybrid model (pytorch-grad-cam).
- LIME image explainer (quickshift superpixels, 500 samples).
- SHAP GradientExplainer on 10 test images (background: 20 train images).
- Output: `outputs/xai/` grid figures (original | Grad-CAM | LIME | SHAP) per image.

## 9. Demo

- `app.py` (Gradio): image upload → class probabilities, Grad-CAM overlay, LIME overlay.
- Loads best hybrid checkpoint.

## 10. Code layout

```
lumpy/
  data/                 raw/, splits.csv (gitignored)
  src/
    config.py           paths, constants
    download.py         Kaggle download + merge
    prepare.py          dedupe + split
    dataset.py          Dataset + transforms
    models.py           HybridNet
    train.py
    evaluate.py
    ablation.py
    xai.py
  app.py
  tests/                smoke tests
  outputs/              runs, figures (gitignored except summaries)
  requirements.txt
```

## 11. Testing

- `tests/test_models.py`: forward pass shapes for all modes/fusions, 2 and 3 classes.
- `tests/test_dataset.py`: transforms output shape/dtype; split has no duplicate hashes
  across splits.

## 12. Risks

- Kaggle slugs may differ from proposal names; resolve by search.
- Very high accuracy may reflect source overlap; dedupe mitigates and the report should
  state it.
- 6 GB VRAM: batch 32 with AMP fits; fall back to batch 16 if OOM.
- SHAP is slow; limited to 10 images.
