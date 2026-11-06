"""API contract tests use CPU doubles so training can keep using the GPU."""
import asyncio
import base64
import io
import json

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


def upload():
    buffer = io.BytesIO()
    Image.new('RGB', (17, 29), 'green').save(buffer, format='PNG')
    return {'file': ('image.png', buffer.getvalue(), 'image/png')}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import api
    runs = tmp_path / 'runs'
    for name, score, classes in [('hybrid_concat_c2', .8, 2), ('hybrid_attention_c2', .9, 2),
                                  ('hybrid_attention_c3', .7, 3)]:
        directory = runs / name
        directory.mkdir(parents=True)
        (directory / 'best.pt').touch()
        (directory / 'config.json').write_text(json.dumps(dict(classes=classes, mode='hybrid', fusion=name.split('_')[1])))
        (directory / 'metrics.json').write_text(json.dumps({'f1_macro': score}))
    monkeypatch.setattr(api, 'RUNS_DIR', runs)
    monkeypatch.setattr(api, 'OUT_DIR', tmp_path)
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    split = tmp_path / 'splits.csv'
    pd.DataFrame([dict(path=str(source), label=i % 3, split=phase)
                  for phase in ['train', 'test'] for i in range(24)]).to_csv(split, index=False)
    monkeypatch.setattr(api, 'SPLITS_CSV', split)
    monkeypatch.setattr(api, 'SPLITS_C3_CSV', split)
    api.models.clear()
    api.backgrounds.clear()
    calls = []

    class Tiny(torch.nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(classes))
        def forward(self, x):
            assert x.device.type == 'cpu'
            assert x.shape[1:] == (3, 224, 224)
            return self.p[None].expand(len(x), -1)

    def load(run):
        calls.append(run)
        classes = 3 if run.endswith('c3') else 2
        return Tiny(classes).eval(), {'classes': classes}

    monkeypatch.setattr(api, 'load_run', load)
    monkeypatch.setattr(api, 'gradcam', lambda model, image: np.zeros((8, 8, 3), dtype=np.uint8))
    def lime(model, image, num_samples):
        assert num_samples == 300
        return np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(api, 'lime_explain', lime)
    def loop_factory():
        # Restricted sandboxes can drop cross-thread socket wakeups. A heartbeat
        # keeps TestClient's real asyncio loop responsive in that environment.
        loop = asyncio.new_event_loop()
        def heartbeat():
            loop.call_later(.01, heartbeat)
        loop.call_soon(heartbeat)
        return loop

    with TestClient(api.app, backend_options={'loop_factory': loop_factory}) as connection:
        yield connection, api, calls


def assert_png(url):
    assert url.startswith('data:image/png;base64,')
    Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))).verify()


def test_models_defaults_lazy(client):
    connection, _, calls = client
    result = connection.get('/api/models')
    assert result.status_code == 200
    assert len(result.json()['runs']) == 3
    assert result.json()['defaults'] == {'2': 'hybrid_attention_c2', '3': 'hybrid_attention_c3'}
    assert calls == []


def test_predict_and_cache(client):
    connection, _, calls = client
    for _ in range(2):
        result = connection.post('/api/predict', files=upload())
        assert result.status_code == 200, result.text
        body = result.json()
        assert body['run'] == 'hybrid_attention_c2'
        assert body['class_names'] == ['healthy', 'lsd']
        assert sum(body['probs'].values()) == pytest.approx(1)
        assert body['confidence'] == max(body['probs'].values())
        assert body['pred'] in body['class_names']
        assert_png(body['gradcam'])
        assert_png(body['lime'])
    assert calls == ['hybrid_attention_c2']


def test_invalid_and_unknown(client):
    connection, _, calls = client
    for endpoint in ['predict', 'shap']:
        result = connection.post('/api/' + endpoint, files={'file': ('bad.png', b'bad', 'image/png')}, data={'run': 'hybrid_attention_c2'})
        assert result.status_code == 400
        assert result.json() == {'detail': 'Invalid image'}
        assert connection.post('/api/' + endpoint, files=upload(), data={'run': '../missing'}).status_code == 404
    assert calls == []


def test_shap_background_cache(client, monkeypatch):
    connection, api, _ = client
    seen = []
    def explain(model, images, background):
        assert len(images) == 1 and len(background) == 20
        seen.append(background)
        return [np.ones((8, 8), dtype=np.float32)]
    monkeypatch.setattr(api, 'shap_explain', explain)
    for _ in range(2):
        result = connection.post('/api/shap', files=upload(), data={'run': 'hybrid_attention_c3'})
        assert result.status_code == 200
        assert_png(result.json()['shap'])
    assert seen[0] is seen[1]


def test_results_samples_and_cors(client):
    connection, api, _ = client
    (api.RUNS_DIR / 'hybrid_attention_c3' / 'confusion_matrix.png').touch()
    pd.DataFrame([{'run': 'example', 'f1': .9}]).to_csv(api.OUT_DIR / 'ablation.csv', index=False)
    body = connection.get('/api/results').json()
    assert body['ablation'] == [{'run': 'example', 'f1': .9}]
    assert body['figures']['hybrid_attention_c3']['confusion'].endswith('/confusion_matrix.png')
    samples = connection.get('/api/samples').json()
    assert len(samples) == 9 and len(set(samples)) == 9
    result = connection.get('/api/models', headers={'Origin': 'http://localhost:5173'})
    assert result.headers['access-control-allow-origin'] == 'http://localhost:5173'
