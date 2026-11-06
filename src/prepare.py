"""Validate images, group perceptual duplicates, and split without group leakage."""
import argparse
from pathlib import Path
import warnings

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, RAW_DIR, SEED, SPLITS_CSV, SPLITS_C3_CSV


def collect(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(raw_dir).rglob('*')):
        if not path.is_file():
            continue
        parts = path.relative_to(raw_dir).parts
        if len(parts) < 3 or parts[1] not in CLASS_NAMES[3]:
            continue
        try:
            with Image.open(path) as image:
                image.convert('RGB').load()
        except (OSError, ValueError) as exc:
            warnings.warn(f'Skipping unreadable image {path}: {exc}', UserWarning)
            continue
        rows.append((str(path.resolve()), CLASS_NAMES[3].index(parts[1]), parts[0]))
    return pd.DataFrame(rows, columns=['path', 'label', 'source'])


def dedupe(df, threshold=4) -> pd.DataFrame:
    result = df.copy()
    parents = list(range(len(df)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    hashes = []
    for path in df.path:
        with Image.open(path) as image:
            hashes.append(int(str(imagehash.phash(image.convert('RGB'))), 16))
    # Exact Hamming comparison ensures no prefix bucket can miss a near duplicate.
    for i, value in enumerate(hashes):
        for j in range(i):
            if (value ^ hashes[j]).bit_count() <= threshold:
                parents[root(i)] = root(j)
    result['group'] = [root(i) for i in range(len(df))]
    return result


def split(df, seed=SEED) -> pd.DataFrame:
    if 'group' not in df:
        raise ValueError('dedupe must add group IDs before splitting')
    result = df.copy()
    if df.empty:
        result['split'] = pd.Series(dtype=str)
        return result
    # Allocate whole groups against both row and per-label targets. Mixed-label
    # duplicate groups remain indivisible. Rare/large groups can limit stratification.
    counts = pd.crosstab(df['group'], df['label'])
    values = np.column_stack([counts.to_numpy(), counts.sum(axis=1).to_numpy()])
    targets = np.array([.7, .15, .15])[:, None] * values.sum(axis=0)
    assigned = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    order = sorted(order, key=lambda i: -values[i, -1])
    mapping = {}
    for i in order:
        delta = (((assigned + values[i] - targets) ** 2 - (assigned - targets) ** 2)
                 / np.maximum(targets, 1)).sum(axis=1)
        choice = int(delta.argmin())
        assigned[choice] += values[i]
        mapping[counts.index[i]] = ('train', 'val', 'test')[choice]
    result['split'] = result['group'].map(mapping)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classes', type=int, choices=[2, 3], default=2)
    args = parser.parse_args(argv)
    df = collect(RAW_DIR)
    df = df[df.label < args.classes]
    if df.empty:
        raise ValueError(f'No readable images found in {RAW_DIR}')
    df = split(dedupe(df))
    destination = SPLITS_CSV if args.classes == 2 else SPLITS_C3_CSV
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    print(df.groupby(['split', 'label']).size().to_string())
    print(f'Duplicate groups: {(df.groupby("group").size() > 1).sum()}; '
          f'redundant images grouped: {len(df) - df.group.nunique()} (all rows retained)')


if __name__ == '__main__':
    main()
