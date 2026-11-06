import numpy as np
import pandas as pd
import torch
from PIL import Image


def test_training_artifacts(tmp_path, monkeypatch):
    from src import train
    from torch.utils.data import DataLoader, TensorDataset
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 2)
        def forward(self, x):
            return self.head(x.mean(dim=(2, 3)))
    dataset = TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1]))
    dataset.df = pd.DataFrame({'label': [0, 1, 0, 1]})
    monkeypatch.setattr(train, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(train, 'build_model', lambda *a, **kw: Tiny())
    monkeypatch.setattr(train, 'make_loaders', lambda *a, **kw: {name: DataLoader(dataset, batch_size=2) for name in ['train', 'val', 'test']})
    train.main(['--mode', 'cnn', '--epochs', '1', '--limit', '4'])
    run = tmp_path / 'cnn_none_c2'
    checkpoint = torch.load(run / 'best.pt', weights_only=True)
    assert checkpoint['mode'] == 'cnn'
    assert len(pd.read_csv(run / 'history.csv')) == 1
    assert (run / 'config.json').is_file()


def test_demo_tiny_image(monkeypatch):
    import app
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            assert x.shape[1:] == (3, 224, 224)
            return torch.zeros(len(x), 2) + self.p
    monkeypatch.setattr(app, 'load_run', lambda run: (Tiny().eval(), {'classes': 2}))
    monkeypatch.setattr(app, 'gradcam', lambda model, image: np.zeros((224, 224, 3), dtype=np.uint8))
    monkeypatch.setattr(app, 'lime_explain', lambda model, image: np.zeros((224, 224, 3), dtype=np.uint8))
    app.configure('test')
    for size in [(100, 300), (1, 2)]:
        probs, cam, lime = app.predict(Image.new('RGB', size))
        assert abs(sum(probs.values()) - 1) < 1e-6
        assert cam.shape == lime.shape == (224, 224, 3)


def test_shap_normalized_inputs(monkeypatch):
    from src.xai import shap_explain
    import shap
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return x.mean((2, 3))[:, :2] + self.p
    class Explainer:
        def __init__(self, model, background):
            assert not model.training
            assert background.shape == (1, 3, 224, 224)
            assert background.min() < 0
        def shap_values(self, images):
            return np.ones((*images.shape, 2))
    monkeypatch.setattr(shap, 'GradientExplainer', Explainer)
    maps = shap_explain(Tiny(), [Image.new('RGB', (3, 5))], [Image.new('RGB', (4, 2))])
    assert len(maps) == 1
    assert maps[0].shape == (224, 224)
    assert np.isfinite(maps[0]).all()


def test_training_limit_missing_class_has_finite_loss(tmp_path, monkeypatch):
    from src import train
    from torch.utils.data import DataLoader, TensorDataset
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 2)
        def forward(self, x):
            return self.head(x.mean((2, 3)))
    loaders = {}
    for name, label in [('train', 0), ('val', 1), ('test', 1)]:
        dataset = TensorDataset(torch.randn(2, 3, 8, 8), torch.full((2,), label))
        dataset.df = pd.DataFrame({'label': [label, label]})
        loaders[name] = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(train, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(train, 'build_model', lambda *a, **kw: Tiny())
    monkeypatch.setattr(train, 'make_loaders', lambda *a, **kw: loaders)
    train.main(['--mode', 'cnn', '--epochs', '1', '--limit', '2'])
    history = pd.read_csv(tmp_path / 'cnn_none_c2' / 'history.csv')
    assert np.isfinite(history[['train_loss', 'val_loss']].to_numpy()).all()


def test_gradcam_real_api():
    from src.xai import gradcam
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, 1)
        @property
        def gradcam_layer(self):
            return self.conv
        def forward(self, x):
            return self.conv(x).mean((2, 3))
    result = gradcam(Tiny().eval(), Image.new('RGB', (5, 19)))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_lime_baseline_matches_prediction(monkeypatch):
    from src.xai import lime_explain
    from src.dataset import get_transforms
    from lime import lime_image
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (31, 7, 3), dtype=np.uint8))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            assert torch.allclose(x[0], get_transforms(False)(image), atol=1e-6)
            return x.mean((2, 3))[:, :2] + self.p
    class Explainer:
        def __init__(self, **kwargs):
            pass
        def explain_instance(self, pixels, classify, **kwargs):
            classify(np.asarray([pixels]))
            self.top_labels = [0]
            return self
        def get_image_and_mask(self, *args, **kwargs):
            return None, np.zeros((224, 224), dtype=int)
    monkeypatch.setattr(lime_image, 'LimeImageExplainer', Explainer)
    assert lime_explain(Tiny().eval(), image).shape == (224, 224, 3)
