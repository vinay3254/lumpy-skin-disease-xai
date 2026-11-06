# LSD Hybrid CNN–Transformer Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and explain a hybrid EfficientNet-B0 + DeiT-Tiny classifier for Lumpy Skin Disease, with ablation, report figures and a Gradio demo.

**Architecture:** Kaggle datasets are downloaded, merged, pHash-deduplicated and split once into `data/splits.csv`. One `HybridNet` class covers all ablation variants via `mode`/`fusion` flags; one training script and one evaluation script operate on named runs under `outputs/runs/`. XAI and the demo load the best hybrid checkpoint.

**Tech Stack:** Python 3, PyTorch (CUDA), timm, torchvision, imagehash, scikit-learn, matplotlib, pytorch-grad-cam (`grad-cam`), lime, shap, gradio, kaggle, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-lsd-hybrid-classifier-design.md`

## Global Constraints

- Image size 224 × 224; ImageNet mean/std normalization; median filter kernel 3.
- Seed 42 everywhere; split 70 / 15 / 15 stratified by label.
- pHash dedupe threshold: Hamming distance ≤ 4.
- Labels: 2-class `{0: healthy, 1: lsd}`; 3-class adds `{2: other}`.
- Batch 32 with AMP (fallback 16 on OOM); max 15 epochs; early stopping on val macro-F1, patience 4.
- AdamW, lr 1e-4 backbones / 3e-4 head, weight decay 0.05, cosine schedule.
- Run name format: `<mode>_<fusion>_c<classes>` (e.g. `hybrid_attention_c2`; `cnn_none_c2`).
- All runnable code under `.venv`; run modules as `python -m src.<name>` from repo root.
- Commit messages: plain, no co-author lines.

## Review Focus

- Grayscale / RGBA / CMYK images in datasets: loader must convert to RGB, not crash (test in Task 2).
- Corrupt or non-image files in downloaded folders: skipped with a warning during prepare (test in Task 2).
- Near-duplicate images across sources: must never land in different splits (test in Task 2).
- Demo receives a non-square or tiny image: preprocessing resizes, prediction still returned (covered by Task 6 smoke test).
- `evaluate`/`xai` called on a run without a checkpoint: clear error message naming the missing path (test in Task 4).

---

### Task 1: Environment, config and model

**Files:**
- Create: `requirements.txt`, `src/__init__.py`, `src/config.py`, `src/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `src.config`: `ROOT`, `DATA_DIR`, `RAW_DIR`, `SPLITS_CSV`, `RUNS_DIR`, `OUT_DIR`, `IMG_SIZE=224`, `SEED=42`, `MEAN`, `STD`, `CLASS_NAMES = {2: ["healthy","lsd"], 3: ["healthy","lsd","other"]}`.
  - `src.models.HybridNet(num_classes: int, mode: str = "hybrid", fusion: str = "attention", pretrained: bool = True)`; `forward(x: Tensor[B,3,224,224]) -> Tensor[B,num_classes]`; attribute `gradcam_layer` (last conv block of CNN branch; `None` for `mode="vit"`).
  - `src.models.build_model(mode, fusion, num_classes, pretrained=True) -> HybridNet`.

- [ ] **Step 1:** Create `.venv`, install torch (CUDA wheel) + `requirements.txt`; verify `python -c "import torch;print(torch.cuda.is_available())"` prints `True`.
- [ ] **Step 2: Write failing tests** in `tests/test_models.py`:

```python
import pytest, torch
from src.models import build_model

@pytest.mark.parametrize("mode,fusion", [("hybrid","concat"),("hybrid","attention"),("cnn","none"),("vit","none")])
@pytest.mark.parametrize("nc", [2, 3])
def test_forward_shape(mode, fusion, nc):
    m = build_model(mode, fusion, nc, pretrained=False).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, nc)

def test_gradcam_layer():
    assert build_model("hybrid","attention",2,pretrained=False).gradcam_layer is not None
    assert build_model("vit","none",2,pretrained=False).gradcam_layer is None
```

- [ ] **Step 3:** Run `pytest tests/test_models.py -q`; expect import failure.
- [ ] **Step 4:** Implement `HybridNet` per spec §5: timm `efficientnet_b0` (`features_only` or `forward_features` → B×1280×7×7), `deit_tiny_patch16_224` `forward_features` → B×197×192; linear projections to 256; `concat` = [pooled CNN ; ViT CLS]; `attention` = `nn.MultiheadAttention(256, 4, batch_first=True)` with CNN tokens as query, ViT tokens as key/value, mean-pooled, concat with ViT CLS; `cnn`/`vit` modes use 256-d pooled vector only, so head input is 256 there and 512 for hybrid. Head: Linear→GELU→Dropout(0.3)→Linear. `gradcam_layer = cnn.conv_head` (or last block).
- [ ] **Step 5:** Run `pytest tests/test_models.py -q`; expect all pass.
- [ ] **Step 6:** Commit `feat: add HybridNet model and project config`.

### Task 2: Data download, dedupe, split and dataset

**Files:**
- Create: `src/download.py`, `src/prepare.py`, `src/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `src.config` paths/constants.
- Produces:
  - `python -m src.download` → images under `data/raw/<source>/<healthy|lsd|other>/`.
  - `src.prepare.collect(raw_dir: Path) -> pd.DataFrame[path,label,source]` (skips unreadable files with a warning).
  - `src.prepare.dedupe(df, threshold=4) -> pd.DataFrame` (adds `group` column; images within threshold share a group; keeps all rows).
  - `src.prepare.split(df, seed=42) -> pd.DataFrame` (adds `split` in {train,val,test}; stratified by label, grouped so a `group` never spans two splits).
  - `python -m src.prepare --classes 2|3` writes `data/splits.csv` for 2 (drops `other`) or `data/splits_c3.csv` for 3; `make_loaders(classes, …)` reads the matching file.
  - `src.dataset.get_transforms(train: bool) -> Callable`; `src.dataset.LSDDataset(df, train: bool)` returns `(Tensor[3,224,224] float32, int)`; `src.dataset.make_loaders(classes: int, batch_size: int) -> dict[str, DataLoader]`.

- [ ] **Step 1:** Write `src/download.py`: search Kaggle (`kaggle datasets list -s "lumpy skin"`), pick the slugs matching spec §3, download/unzip to `data/tmp/`, map folder names to `healthy|lsd` (and `other` for non-LSD disease classes) into `data/raw/<source>/`. Hard-code chosen slugs and folder mappings in a `SOURCES` dict after inspection.
- [ ] **Step 2: Write failing tests** in `tests/test_dataset.py`:
  - `test_collect_skips_corrupt(tmp_path)`: writes one valid JPEG + one text file named `.jpg` under `src1/lsd/`; `len(collect(tmp_path)) == 1`.
  - `test_dedupe_groups_duplicates(tmp_path)`: same image saved twice (one slightly resized) + one different image; dedupe gives 2 distinct groups.
  - `test_split_no_group_leak`: synthetic df of 100 rows/30 groups; each group appears in exactly one split; split proportions within ±10 % of 70/15/15.
  - `test_dataset_rgb_conversion(tmp_path)`: grayscale and RGBA images load as shape `(3,224,224)`, dtype float32.
- [ ] **Step 3:** Run `pytest tests/test_dataset.py -q`; expect failures.
- [ ] **Step 4:** Implement `prepare.py` (imagehash.phash; union-find over pairs with distance ≤ 4 — bucket by hash prefix or brute-force numpy XOR popcount, fine for ≤ 15k images; `StratifiedGroupKFold`-style split via sklearn `GroupShuffleSplit` per label or `StratifiedGroupKFold(n_splits=...)`) and `dataset.py` (PIL `.convert("RGB")`, `ImageFilter.MedianFilter(3)`, transforms per spec §4, `WeightedRandomSampler` not used — class weights handled in loss).
- [ ] **Step 5:** Run `pytest tests/test_dataset.py -q`; expect pass.
- [ ] **Step 6:** Run `python -m src.download && python -m src.prepare --classes 2`; print per-split per-label counts and number of removed-duplicate groups. Expect non-empty train/val/test for both labels.
- [ ] **Step 7:** Commit `feat: add data download, dedupe, split and dataset`.

### Task 3: Training script

**Files:**
- Create: `src/train.py`

**Interfaces:**
- Consumes: `build_model`, `make_loaders`.
- Produces: `python -m src.train --mode {hybrid,cnn,vit} --fusion {concat,attention,none} --classes {2,3} [--epochs 15 --batch 32 --limit N]` writing `outputs/runs/<run_name>/best.pt` (dict with `state_dict`, `mode`, `fusion`, `classes`), `history.csv` (epoch, train_loss, val_loss, train_acc, val_acc, val_f1), `config.json`. `--limit N` truncates each split to N items (smoke runs).

- [ ] **Step 1:** Implement per spec §6: param groups (backbones 1e-4, rest 3e-4), AdamW wd 0.05, `CosineAnnealingLR`, `torch.autocast` + `GradScaler`, class-weighted `CrossEntropyLoss` from train label counts, early stopping on val macro-F1 patience 4, seed 42.
- [ ] **Step 2:** Smoke run: `python -m src.train --mode hybrid --fusion attention --classes 2 --epochs 1 --limit 64`. Expect run dir with `best.pt`, `history.csv` (1 row), `config.json`.
- [ ] **Step 3:** Commit `feat: add training script`.

### Task 4: Evaluation and ablation

**Files:**
- Create: `src/evaluate.py`, `src/ablation.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: run dir from Task 3.
- Produces:
  - `src.evaluate.load_run(run: str) -> tuple[HybridNet, dict]`; raises `FileNotFoundError` with message containing the missing `best.pt` path.
  - `python -m src.evaluate --run <name>` writes `metrics.json` (accuracy, precision_macro, recall_macro, f1_macro, roc_auc), `confusion_matrix.png`, `roc_curve.png`, `curves.png`, `classification_report.txt` into the run dir.
  - `python -m src.ablation` writes `outputs/ablation.csv` and `outputs/ablation.md` (one row per run, columns: run, params_M, accuracy, precision, recall, f1, roc_auc).

- [ ] **Step 1: Write failing test** `test_load_run_missing`: `with pytest.raises(FileNotFoundError, match="best.pt"): load_run("does_not_exist")`.
- [ ] **Step 2:** Run test; expect fail.
- [ ] **Step 3:** Implement both modules. ROC-AUC: binary uses positive-class prob; 3-class uses `roc_auc_score(..., multi_class="ovr")`, ROC plot one curve per class.
- [ ] **Step 4:** Run test; expect pass. Run `python -m src.evaluate --run hybrid_attention_c2` on the smoke checkpoint; expect files listed above.
- [ ] **Step 5:** Commit `feat: add evaluation and ablation summary`.

### Task 5: Explainability

**Files:**
- Create: `src/xai.py`

**Interfaces:**
- Consumes: `load_run`, `get_transforms`, `SPLITS_CSV`.
- Produces:
  - `src.xai.gradcam(model, img: PIL.Image) -> np.ndarray[H,W,3] uint8` overlay.
  - `src.xai.lime_explain(model, img: PIL.Image, num_samples=500) -> np.ndarray[H,W,3] uint8` boundary overlay.
  - `src.xai.shap_explain(model, imgs: list[PIL.Image], background: list[PIL.Image]) -> list[np.ndarray]` heatmaps.
  - `python -m src.xai --run <name> --n 10` writes `outputs/xai/<run>/sample_XX.png` (grid: original | Grad-CAM | LIME | SHAP, title with true/pred label and confidence) for n test images (balanced across classes).

- [ ] **Step 1:** Implement with `pytorch_grad_cam.GradCAM(model, [model.gradcam_layer])`, `lime.lime_image.LimeImageExplainer` (quickshift, top 5 superpixels, positive only), `shap.GradientExplainer` with 20 train background images; SHAP map = sum abs over channels, upsampled to 224.
- [ ] **Step 2:** Run `python -m src.xai --run hybrid_attention_c2 --n 2` on smoke checkpoint; expect 2 PNGs with 4 panels.
- [ ] **Step 3:** Commit `feat: add Grad-CAM, LIME and SHAP explanations`.

### Task 6: Gradio demo

**Files:**
- Create: `app.py`, `README.md`

**Interfaces:**
- Consumes: `load_run`, `gradcam`, `lime_explain`, `get_transforms`.
- Produces: `python app.py [--run hybrid_attention_c2]` launches Gradio: input image → `gr.Label` (class probs), Grad-CAM image, LIME image.

- [ ] **Step 1:** Implement `app.py` with a `predict(img: PIL.Image) -> tuple[dict, np.ndarray, np.ndarray]` function used by the Interface.
- [ ] **Step 2:** Smoke: call `predict` from Python on a 100×300 test image; expect probs summing to 1 and two (224,224,3) arrays.
- [ ] **Step 3:** Write `README.md`: setup, commands for download → prepare → train → evaluate → ablation → xai → app.
- [ ] **Step 4:** Commit `feat: add Gradio demo and README`.

### Task 7: Full experiments (binary)

- [ ] **Step 1:** Train 4 runs: `hybrid/attention`, `hybrid/concat`, `cnn/none`, `vit/none`, all `--classes 2`.
- [ ] **Step 2:** `python -m src.evaluate --run <each>`; then `python -m src.ablation`.
- [ ] **Step 3:** `python -m src.xai --run <best hybrid> --n 10`.
- [ ] **Step 4:** Commit summaries (`outputs/ablation.*`, run `metrics.json` and PNGs; checkpoints stay ignored) `exp: binary results, ablation and XAI figures`.

### Task 8 (stretch): 3-class extension

- [ ] **Step 1:** Only if `other` images exist in `data/raw`: `python -m src.prepare --classes 3` (writes `data/splits_c3.csv`; `make_loaders(3, …)` reads it), train `hybrid/attention --classes 3`, evaluate, XAI n=6.
- [ ] **Step 2:** Commit `exp: 3-class extension results`. If no `other` data, record in README "Future work".
