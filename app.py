"""Gradio image classifier with Grad-CAM and LIME overlays."""
import argparse

import numpy as np
from PIL import Image
import torch

from src.config import CLASS_NAMES
from src.dataset import get_transforms
from src.evaluate import load_run
from src.xai import gradcam, lime_explain

_model = None
_config = None
_run = 'hybrid_attention_c2'


def configure(run):
    global _model, _config, _run
    _run = run
    _model = _config = None


def predict(img: Image.Image) -> tuple[dict, np.ndarray, np.ndarray]:
    global _model, _config
    if img is None:
        raise ValueError('Please upload an image')
    if _model is None:
        _model, _config = load_run(_run)
        if getattr(_model, 'gradcam_layer', True) is None:
            _model = None
            raise ValueError('The demo requires a CNN or hybrid checkpoint for Grad-CAM')
    image = img.convert('RGB')
    tensor = get_transforms(False)(image).unsqueeze(0).to(next(_model.parameters()).device)
    _model.eval()
    with torch.inference_mode():
        probabilities = _model(tensor).softmax(1)[0].cpu().tolist()
    return (dict(zip(CLASS_NAMES[_config['classes']], probabilities)),
            gradcam(_model, image), lime_explain(_model, image))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='hybrid_attention_c2')
    args = parser.parse_args(argv)
    import gradio as gr
    configure(args.run)
    gr.Interface(fn=predict, inputs=gr.Image(type='pil', label='Cattle image'),
                 outputs=[gr.Label(label='Class probabilities'), gr.Image(label='Grad-CAM'), gr.Image(label='LIME')],
                 title='Lumpy Skin Disease Classifier',
                 description='Upload a cattle image to view the prediction and visual explanations.').launch()


if __name__ == '__main__':
    main()
