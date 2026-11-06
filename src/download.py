"""Download the configured Kaggle sources and normalize their folder labels."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
from collections import Counter

from src.config import DATA_DIR, RAW_DIR

SOURCES = {'warcoder': 'warcoder/lumpy-skin-images-dataset',
           'cattle_diseases': 'devang03mgr/cattle-diseases-datasets'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def map_label(path_parts) -> str | None:
    """Use the nearest recognized parent; specific child folders override ancestors."""
    for part in reversed(list(path_parts)):
        name = re.sub(r'[^a-z0-9]+', ' ', str(part).lower())
        if any(word in name for word in ('normal', 'healthy')):
            return 'healthy'
        if any(word in name for word in ('lumpy', 'lsd', 'infected')):
            return 'lsd'
        if any(word in name for word in ('other', 'foot', 'mouth', 'fmd', 'ringworm', 'ibk', 'pinkeye', 'dermat', 'mastitis')):
            return 'other'
    return None


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    # Lazy import: --help and module imports never authenticate or access the network.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    for source, slug in SOURCES.items():
        folder = DATA_DIR / 'tmp' / slug.split('/')[-1]
        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(slug, path=str(folder), unzip=True)
        counts = Counter()
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(folder)
            label = map_label(relative.parts[:-1])
            if label is None:
                print(f'Skipping unmapped folder: {path.parent}')
                continue
            destination = RAW_DIR / source / label
            destination.mkdir(parents=True, exist_ok=True)
            unique = hashlib.sha256(str(relative).encode()).hexdigest()
            shutil.copy2(path, destination / f'{unique}{path.suffix.lower()}')
            counts[label] += 1
        for label in ('healthy', 'lsd', 'other'):
            print(f'{source}/{label}: {counts[label]}')


if __name__ == '__main__':
    main()
