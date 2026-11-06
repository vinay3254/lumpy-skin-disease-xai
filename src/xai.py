"""Grad-CAM, LIME and SHAP explanations for a saved classifier run."""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.config import CLASS_NAMES, IMG_SIZE, MEAN, STD, OUT_DIR, SEED, SPLITS_CSV, SPLITS_C3_CSV
from src.dataset import get_transforms
from src.evaluate import load_run


def _rgb(img):
    return img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))


def _tensor(model, images):
    return torch.stack([get_transforms(False)(image) for image in images]).to(next(model.parameters()).device)


def gradcam(model, img: Image.Image) -> np.ndarray:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    if model.gradcam_layer is None:
        raise ValueError('Grad-CAM requires a CNN branch; this run uses ViT only')
    model.eval()
    image = _rgb(img)
    with torch.enable_grad():
        with GradCAM(model=model, target_layers=[model.gradcam_layer]) as cam:
            heatmap = cam(input_tensor=_tensor(model, [img]), targets=None)[0]
    return show_cam_on_image(np.asarray(image, dtype=np.float32) / 255., heatmap, use_rgb=True)


def lime_explain(model, img: Image.Image, num_samples=500) -> np.ndarray:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    model.eval()
    device = next(model.parameters()).device
    mean = torch.tensor(MEAN, device=device)[None, :, None, None]
    std = torch.tensor(STD, device=device)[None, :, None, None]
    # Segment the exact median-filtered/resized image used for prediction.
    baseline = _tensor(model, [img])
    image = ((baseline * std + mean)[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)

    def classify(batch):
        tensor = torch.as_tensor(np.asarray(batch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
        with torch.inference_mode():
            return model((tensor - mean) / std).softmax(1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    explanation = explainer.explain_instance(
        image, classify, top_labels=1, hide_color=0, num_samples=num_samples,
        segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=.2),
        random_seed=SEED)
    _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,
                                              num_features=5, hide_rest=False)
    return (mark_boundaries(image / 255., mask) * 255).clip(0, 255).astype(np.uint8)


def shap_explain(model, imgs: list[Image.Image], background: list[Image.Image]) -> list[np.ndarray]:
    import shap
    if not imgs:
        return []
    if not background:
        raise ValueError('SHAP requires at least one background image')
    model.eval()
    images = _tensor(model, imgs)
    reference = _tensor(model, background)
    with torch.enable_grad():
        explainer = shap.GradientExplainer(model, reference, batch_size=8)
        # One image at a time with few samples keeps SHAP within 6 GB VRAM.
        parts = [explainer.shap_values(images[i:i + 1], nsamples=50) for i in range(len(images))]
        values = ([np.concatenate([p[k] for p in parts]) for k in range(len(parts[0]))]
                  if isinstance(parts[0], list) else np.concatenate(parts))
    with torch.inference_mode():
        predicted = model(images).argmax(1).cpu().numpy()
    # Older SHAP: list of [N,C,H,W]. Newer SHAP: [N,C,H,W,outputs].
    if isinstance(values, list):
        selected = np.stack([np.asarray(values[int(label)])[i] for i, label in enumerate(predicted)])
    else:
        values = np.asarray(values)
        selected = np.stack([values[i, ..., int(label)] for i, label in enumerate(predicted)]) if values.ndim == 5 else values
    from scipy.ndimage import gaussian_filter
    # Smooth away the ViT 16x16 patch-grid artefact so regions read clearly.
    heatmaps = [gaussian_filter(h, sigma=6) for h in np.abs(selected).sum(axis=1)]
    return [np.asarray(Image.fromarray(heat.astype(np.float32)).resize((IMG_SIZE, IMG_SIZE))) for heat in heatmaps]


def _balanced_samples(df, n):
    groups = [group.sample(frac=1, random_state=SEED).reset_index(drop=True)
              for _, group in df.groupby('label')]
    rows = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                rows.append(group.iloc[index])
                if len(rows) == n:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=df.columns)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error('--n must be positive')
    model, config = load_run(args.run)
    if model.gradcam_layer is None:
        raise ValueError('The four-panel XAI grid requires a hybrid or CNN run')
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    df = pd.read_csv(SPLITS_CSV if config['classes'] == 2 else SPLITS_C3_CSV)
    selected = _balanced_samples(df[df['split'] == 'test'], args.n)
    training = df[df['split'] == 'train'].sample(frac=1, random_state=SEED).head(20)
    if selected.empty or training.empty:
        raise ValueError('XAI requires nonempty train and test splits')

    def open_rgb(path):
        with Image.open(path) as image:
            return image.convert('RGB')

    images = [open_rgb(path) for path in selected.path]
    background = [open_rgb(path) for path in training.path]
    heats = shap_explain(model, images, background)
    destination = OUT_DIR / 'xai' / args.run
    destination.mkdir(parents=True, exist_ok=True)
    names = CLASS_NAMES[config['classes']]
    for index, (image, heat, label) in enumerate(zip(images, heats, selected.label)):
        with torch.inference_mode():
            probs = model(_tensor(model, [image])).softmax(1)[0].cpu().numpy()
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [_rgb(image), gradcam(model, image), lime_explain(model, image), heat]
        for ax, panel, title in zip(axes, panels, ['Original', 'Grad-CAM', 'LIME', 'SHAP']):
            if title == 'SHAP':
                ax.imshow(panels[0])
                ax.imshow(panel, cmap='inferno', alpha=0.6)
            else:
                ax.imshow(panel)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'True: {names[int(label)]} | Pred: {names[int(probs.argmax())]} ({probs.max():.1%})')
        fig.tight_layout()
        fig.savefig(destination / f'sample_{index:02d}.png', bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
