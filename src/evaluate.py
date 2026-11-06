"""Evaluate a saved run and create report figures on its held-out test split."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay,
                             classification_report)

from src.config import CLASS_NAMES, RUNS_DIR
from src.dataset import make_loaders
from src.models import build_model


def load_run(run: str):
    path = RUNS_DIR / run / 'best.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['mode'], checkpoint['fusion'], checkpoint['classes'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return model, {key: checkpoint[key] for key in ('mode', 'fusion', 'classes')}


def compute_metrics(truth, probabilities):
    classes = probabilities.shape[1]
    predictions = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=list(range(classes)), average='macro', zero_division=0)
    try:
        auc = roc_auc_score(truth, probabilities[:, 1]) if classes == 2 else roc_auc_score(
            truth, probabilities, labels=list(range(classes)), multi_class='ovr')
        auc = float(auc) if np.isfinite(auc) else None
    except ValueError:
        auc = None  # A tiny test split may not contain every class.
    return dict(accuracy=float(accuracy_score(truth, predictions)), precision_macro=float(precision),
                recall_macro=float(recall), f1_macro=float(f1), roc_auc=auc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args(argv)
    model, config = load_run(args.run)
    device = next(model.parameters()).device
    truth, probabilities = [], []
    with torch.inference_mode():
        for images, labels in make_loaders(config['classes'], 32)['test']:
            truth.extend(labels.tolist())
            probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
    if not truth:
        raise ValueError('Test split contains no images')
    truth, probabilities = np.asarray(truth), np.concatenate(probabilities)
    predictions = probabilities.argmax(1)
    names = CLASS_NAMES[config['classes']]
    run_dir = RUNS_DIR / args.run
    metrics = compute_metrics(truth, probabilities)
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (run_dir / 'classification_report.txt').write_text(classification_report(
        truth, predictions, labels=list(range(len(names))), target_names=names, zero_division=0))
    display = ConfusionMatrixDisplay.from_predictions(truth, predictions, labels=list(range(len(names))), display_labels=names)
    display.figure_.savefig(run_dir / 'confusion_matrix.png', bbox_inches='tight')
    plt.close(display.figure_)
    fig, ax = plt.subplots()
    for i in ([1] if len(names) == 2 else range(len(names))):
        binary = truth == i
        if binary.any() and not binary.all():
            fpr, tpr, _ = roc_curve(binary, probabilities[:, i])
            ax.plot(fpr, tpr, label=names[i])
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC')
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.savefig(run_dir / 'roc_curve.png', bbox_inches='tight')
    plt.close(fig)
    history = pd.read_csv(run_dir / 'history.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for phase in ('train', 'val'):
        axes[0].plot(history.epoch, history[f'{phase}_loss'], label=phase)
        axes[1].plot(history.epoch, history[f'{phase}_acc'], label=phase)
    for ax, title in zip(axes, ('Loss', 'Accuracy')):
        ax.set(xlabel='Epoch', title=title)
        ax.legend()
    fig.savefig(run_dir / 'curves.png', bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
