"""Image loading and ImageNet preprocessing shared by training and inference."""
import argparse
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import IMG_SIZE, MEAN, STD, SEED, SPLITS_CSV, SPLITS_C3_CSV


def get_transforms(train: bool):
    operations = [transforms.Lambda(lambda img: img.convert('RGB').filter(ImageFilter.MedianFilter(3)))]
    if train:
        operations += [transforms.RandomResizedCrop(IMG_SIZE, scale=(.8, 1.0)),
                       transforms.RandomRotation(20), transforms.RandomHorizontalFlip(),
                       transforms.ColorJitter(brightness=.2, contrast=.2)]
    else:
        operations += [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class LSDDataset(Dataset):
    def __init__(self, df, train: bool):
        self.df = df.reset_index(drop=True).copy()
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image)
        return tensor, int(row.label)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(classes: int, batch_size: int, limit=None):
    if classes not in (2, 3):
        raise ValueError('classes must be 2 or 3')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    df = pd.read_csv(SPLITS_CSV if classes == 2 else SPLITS_C3_CSV)
    loaders = {}
    for name in ('train', 'val', 'test'):
        subset = df[df['split'] == name]
        if limit is not None:
            subset = subset.iloc[:limit]
        loaders[name] = DataLoader(LSDDataset(subset, train=name == 'train'),
                                   batch_size=batch_size, shuffle=name == 'train' and len(subset) > 0,
                                   num_workers=4, worker_init_fn=seed_worker,
                                   generator=torch.Generator().manual_seed(SEED))
    return loaders


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)


if __name__ == '__main__':
    main()
