# Lumpy Skin Disease classifier

EfficientNet-B0 and DeiT-Tiny combine local and global image features. The same
model supports CNN-only, ViT-only, concatenation, and cross-attention ablations.
Binary labels are `healthy` and `lsd`; the optional third class is `other`.

Use the existing `.venv` from the repository root. Dependencies are listed in
`requirements.txt`; PyTorch, torchvision and CUDA must be compatible. Downloads
require network access and Kaggle credentials configured for the Kaggle API.
Training with pretrained backbones requires cached timm weights or network
access. Loading a saved run never downloads weights.

```bash
.venv/bin/python -m src.download
.venv/bin/python -m src.prepare --classes 2
.venv/bin/python -m src.train --mode hybrid --fusion attention --classes 2
.venv/bin/python -m src.evaluate --run hybrid_attention_c2
.venv/bin/python -m src.ablation
.venv/bin/python -m src.xai --run hybrid_attention_c2 --n 10
.venv/bin/python app.py --run hybrid_attention_c2
```

Download sources are `warcoder/lumpy-skin-images-dataset` and
`devang03mgr/cattle-diseases-datasets`. Existing `data/tmp/<slug-name>/`
directories are reused. Adjust `map_label` in `src/download.py` to customize
folder mapping. Images are copied to `data/raw/<source>/<label>/`; unmatched
folders are printed. Preparation warns and skips unreadable files, groups
pHash matches within Hamming distance 4 transitively across all sources, and
keeps every image while assigning each whole group to one split. Targets are
70/15/15 stratified by label, seed 42. Large duplicate groups or rare labels
can prevent exact proportions; inspect the printed split counts. Grouping
prevents duplicate leakage but repeated images still affect class frequencies.

Images are converted to RGB, median-filtered with kernel 3, resized to 224×224,
and normalized with ImageNet statistics. Rotation, flip, crop and color jitter
apply only to training. Loaders use four workers. Training uses weighted cross
entropy, AdamW, cosine scheduling, CUDA AMP where available, at most 15 epochs,
and early stopping on validation macro-F1 with patience 4. CUDA OOM restarts
training with batch 16 when the requested batch exceeds 16.

Runs are named `<mode>_<fusion>_c<classes>` under `outputs/runs/` and contain
`best.pt`, `history.csv` and `config.json`. Reusing a run name replaces its
training artifacts. Rerun evaluation after retraining: previous metrics and
figures remain until evaluation replaces them, and ablation reads those metrics.
Evaluation adds metrics, a classification report, confusion
matrix, ROC and training curves. ROC-AUC is JSON `null` when a test split lacks
required classes. Ablation summarizes evaluated runs into `outputs/ablation.csv`
and `.md`.

For the four binary ablations, train and evaluate these configurations:

```bash
.venv/bin/python -m src.train --mode cnn --classes 2
.venv/bin/python -m src.train --mode vit --classes 2
.venv/bin/python -m src.train --mode hybrid --fusion concat --classes 2
.venv/bin/python -m src.train --mode hybrid --fusion attention --classes 2
```

Non-hybrid runs use fusion `none`. `--epochs 1 --limit 64` truncates each split
for a smoke run. For the optional extension, prepare with `--classes 3` to create
`data/splits_c3.csv`, then train/evaluate `hybrid_attention_c3`. Complete binary
experiments before the extension. Severity grading remains future work.

XAI grids go to `outputs/xai/<run>/sample_XX.png` with original, Grad-CAM, LIME
(500 samples) and SHAP panels. SHAP uses up to 20 training background images and
explains the predicted class; test selections alternate classes. Grad-CAM and
the demo require a CNN branch. The demo loads its checkpoint on first prediction
and accepts tiny and non-square images.

```bash
.venv/bin/python -m pytest -q
```

Tests use temporary image files and synthetic inputs; they do not download,
prepare real datasets, train full models, or establish clinical performance.
