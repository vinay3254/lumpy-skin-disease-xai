# User Manual — LSD-XAI

## 1. Requirements

| Item | Version |
|---|---|
| OS | Linux (tested on Arch Linux) |
| Python | 3.12 |
| Node.js | 20 or newer (tested on 26) |
| GPU | NVIDIA GPU with CUDA 12.4 drivers, 6 GB VRAM or more (CPU works, but slower) |
| Kaggle | API token in `~/.kaggle/` (only to download the datasets) |

## 2. Installation

```bash
git clone https://github.com/vinay3254/lumpy-skin-disease-xai.git
cd lumpy-skin-disease-xai

# Python environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt

# Web app
cd frontend && npm install && cd ..
```

## 3. Data and training

Trained model files (`best.pt`) are not in the repository because of their size.
To rebuild them:

```bash
.venv/bin/python -m src.download              # downloads about 430 MB from Kaggle
.venv/bin/python -m src.prepare --classes 2
.venv/bin/python -m src.prepare --classes 3
.venv/bin/python -m src.train --mode hybrid --fusion attention --classes 3
.venv/bin/python -m src.train --mode hybrid --fusion concat --classes 2 \
    --backbone-lr 3e-5 --label-smoothing 0.1 --patience 6 --tag _v2
.venv/bin/python -m src.evaluate --run hybrid_attention_c3
.venv/bin/python -m src.evaluate --run hybrid_concat_c2_v2
```

Each training run takes about 5–10 minutes on an RTX 4050.

## 4. Running the web app

```bash
./start.sh
```

Open http://localhost:5173. Stop with `Ctrl+C`.

## 5. Using the web app

### Diagnose
1. Drop an image on the upload box, click **Browse images**, or click a sample.
2. Select **Binary** (Healthy / LSD) or **3-class** (Healthy / LSD / Other).
3. Click **Analyze image**.
4. Read the predicted class, the confidence and the per-class probabilities.
5. Check the Grad-CAM and LIME explanations below the result.
6. Click **Explain deeper (SHAP)** for the SHAP map (about 5 seconds).

### Results
- Ablation table of all trained runs; the best F1 run is highlighted.
- Accuracy and F1 chart.
- Select a run to see its confusion matrix, ROC curve and training curves.

### About
- Pipeline, architecture, datasets and the leakage-prevention method.

## 6. Command-line tools

| Command | Purpose |
|---|---|
| `python -m src.evaluate --run <run>` | Test metrics and figures for a run |
| `python -m src.ablation` | Builds `outputs/ablation.csv` and `.md` |
| `python -m src.xai --run <run> --n 10` | Saves XAI figures to `outputs/xai/<run>/` |
| `python app.py --run <run>` | Backup Gradio demo |
| `python -m pytest -q` | Runs the 33 tests |

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Close other GPU programs. Training falls back to batch 16 on its own. |
| Web app shows an error on Analyze | Check that the API runs on port 8000: `curl localhost:8000/api/models`. |
| Port 8000 or 5173 already in use | Stop the old process, then run `./start.sh` again. |
| `FileNotFoundError ... best.pt` | The run is not trained yet; see Section 3. |
| First prediction is slow | The model is loading into GPU memory; later predictions are fast. |

## 8. Disclaimer

LSD-XAI is a research prototype. It is not a veterinary diagnostic tool.
Consult a qualified veterinarian for animal health decisions.
