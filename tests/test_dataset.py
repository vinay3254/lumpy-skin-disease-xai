import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import LSDDataset
from src.prepare import collect, dedupe, split
from src.download import map_label


def test_collect_skips_corrupt(tmp_path):
    folder = tmp_path / 'src1' / 'lsd'
    folder.mkdir(parents=True)
    Image.new('RGB', (32, 32)).save(folder / 'valid.jpg')
    (folder / 'bad.jpg').write_text('not an image')
    (folder / 'notes.txt').write_text('notes')
    with pytest.warns(UserWarning):
        df = collect(tmp_path)
    assert len(df) == 1
    assert df.iloc[0].label == 1
    assert df.iloc[0].source == 'src1'


def test_dedupe_groups_duplicates(tmp_path):
    rng = np.random.default_rng(42)
    first = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    second = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    paths = [tmp_path / f'{i}.png' for i in range(3)]
    first.save(paths[0])
    first.resize((72, 72)).save(paths[1])
    second.save(paths[2])
    df = dedupe(pd.DataFrame({'path': paths, 'label': [0, 0, 1], 'source': ['a', 'b', 'c']}))
    assert df.group.nunique() == 2
    assert df.group.iloc[0] == df.group.iloc[1]


def test_split_no_group_leak():
    groups = np.arange(100) % 30
    df = pd.DataFrame({'path': [str(i) for i in range(100)], 'group': groups, 'label': groups % 2})
    result = split(df)
    assert result.groupby('group')['split'].nunique().max() == 1
    for name, expected in [('train', .7), ('val', .15), ('test', .15)]:
        assert abs((result['split'] == name).mean() - expected) <= .1
    pd.testing.assert_frame_equal(result, split(df))


@pytest.mark.parametrize('mode', ['L', 'RGBA', 'CMYK'])
def test_dataset_rgb_conversion(tmp_path, mode):
    path = tmp_path / ('image.jpg' if mode == 'CMYK' else 'image.png')
    Image.new(mode, (7, 31)).save(path)
    image, label = LSDDataset(pd.DataFrame({'path': [path], 'label': [1]}), train=False)[0]
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label == 1


@pytest.mark.parametrize('parts,label', [(['dataset', 'Lumpy Skin'], 'lsd'), (['LSD', 'Normal'], 'healthy'), (['dataset', 'INFECTED'], 'lsd'), (['data', 'Foot and Mouth'], 'other'), (['data', 'unknown'], None)])
def test_map_label(parts, label):
    assert map_label(parts) == label
