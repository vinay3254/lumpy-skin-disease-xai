"""Local demo API; reuse the training pipeline's preprocessing and explanations."""
import base64
from contextlib import asynccontextmanager
import io
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch

from src.config import CLASS_NAMES, OUT_DIR, RUNS_DIR, SEED, SPLITS_CSV, SPLITS_C3_CSV
from src.dataset import get_transforms
from src.evaluate import load_run
from src.xai import gradcam, lime_explain, shap_explain

models = {}
backgrounds = {}
gpu_lock = threading.Lock()


def run_catalog():
    runs = []
    for directory in sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []:
        if not (directory / 'best.pt').is_file() or not (directory / 'metrics.json').is_file():
            continue
        config_path = directory / 'config.json'
        if config_path.is_file():
            config = json.loads(config_path.read_text())
        else:
            # Legacy runs may only store configuration in their checkpoint.
            config = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        runs.append(dict(run=directory.name, **{key: config[key] for key in ('classes', 'mode', 'fusion')},
                         metrics=json.loads((directory / 'metrics.json').read_text())))
    binary = [row for row in runs if row['classes'] == 2 and row['mode'] == 'hybrid' and '_c2' in row['run']]
    best = max(binary, key=lambda row: row['metrics'].get('f1_macro', row['metrics'].get('f1', 0)) or 0, default=None)
    return {'runs': runs, 'defaults': {'2': best['run'] if best else None,
                                     '3': 'hybrid_attention_c3' if any(row['run'] == 'hybrid_attention_c3' for row in runs) else None}}


def resolve_run(run):
    catalog = run_catalog()
    selected = run if run is not None else catalog['defaults']['2']
    if selected is None or selected not in {row['run'] for row in catalog['runs']}:
        raise HTTPException(404, 'Unknown run')
    return selected


def get_model(run):
    # Called only under gpu_lock, including the initial device transfer.
    if run not in models:
        models[run] = load_run(run)
    return models[run]


def read_image(file):
    try:
        with Image.open(file.file) as image:
            return image.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, 'Invalid image') from None


def png_url(pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        pixels = np.nan_to_num(pixels).astype(np.float64)
        low, high = pixels.min(), pixels.max()
        pixels = ((pixels - low) / (high - low) * 255 if high > low else np.zeros_like(pixels)).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def prepare_samples():
    destination = OUT_DIR / 'samples'
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()) or not SPLITS_C3_CSV.is_file():
        return
    df = pd.read_csv(SPLITS_C3_CSV)
    for label, group in df[df['split'] == 'test'].groupby('label'):
        for index, path in enumerate(group.head(3).path):
            source = Path(path)
            shutil.copyfile(source, destination / f'{int(label)}_{index}{source.suffix.lower()}')


@asynccontextmanager
async def lifespan(app):
    prepare_samples()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'],
                   allow_methods=['GET', 'POST'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory=OUT_DIR, check_dir=False), name='static')


@app.get('/api/models')
def list_models():
    return run_catalog()


@app.post('/api/predict')
def predict(file: UploadFile = File(...), run: str | None = Form(None)):
    image = read_image(file)
    run = resolve_run(run)
    with gpu_lock:
        model, config = get_model(run)
        names = CLASS_NAMES[config['classes']]
        tensor = get_transforms(False)(image).unsqueeze(0).to(next(model.parameters()).device)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(1)[0].cpu().tolist()
        try:
            cam = png_url(gradcam(model, image))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        lime = png_url(lime_explain(model, image, num_samples=300))
    return dict(run=run, class_names=names, probs=dict(zip(names, probabilities)),
                pred=names[int(np.argmax(probabilities))], confidence=max(probabilities), gradcam=cam, lime=lime)


@app.post('/api/shap')
def explain_shap(file: UploadFile = File(...), run: str = Form(...)):
    image = read_image(file)
    run = resolve_run(run)
    with gpu_lock:
        model, config = get_model(run)
        if run not in backgrounds:
            df = pd.read_csv(SPLITS_CSV if config['classes'] == 2 else SPLITS_C3_CSV)
            training = df[df['split'] == 'train'].sample(frac=1, random_state=SEED).head(20)
            if training.empty:
                raise HTTPException(422, 'SHAP requires training images')
            images = []
            for path in training.path:
                with Image.open(path) as reference:
                    images.append(reference.convert('RGB'))
            backgrounds[run] = images
        heat = shap_explain(model, [image], backgrounds[run])[0]
        return {'shap': png_url(heat)}


@app.get('/api/results')
def results():
    ablation = OUT_DIR / 'ablation.csv'
    rows = json.loads(pd.read_csv(ablation).to_json(orient='records')) if ablation.is_file() else []
    figures = {}
    for directory in sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []:
        if directory.is_dir():
            panels = {key: f'/static/runs/{quote(directory.name)}/{filename}'
                      for key, filename in [('confusion', 'confusion_matrix.png'), ('roc', 'roc_curve.png'), ('curves', 'curves.png')]
                      if (directory / filename).is_file()}
            if panels:
                figures[directory.name] = panels
    return {'ablation': rows, 'figures': figures}


@app.get('/api/samples')
def samples():
    destination = OUT_DIR / 'samples'
    return [f'/static/samples/{quote(path.name)}' for path in sorted(destination.iterdir())
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}] if destination.exists() else []
