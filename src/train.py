"""Train a named classifier run with AMP and validation-F1 early stopping."""
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.config import RUNS_DIR, SEED
from src.dataset import make_loaders
from src.models import build_model


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(args, batch_size):
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = make_loaders(args.classes, batch_size, limit=args.limit)
    if any(len(loaders[name].dataset) == 0 for name in ('train', 'val')):
        raise ValueError('Training and validation splits must both contain images')
    model = build_model(args.mode, args.fusion, args.classes).to(device)
    backbone, head = [], []
    for name, param in model.named_parameters():
        (backbone if name.startswith(('cnn.', 'vit.')) else head).append(param)
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.backbone_lr},
                                   {'params': head, 'lr': 3e-4}], weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    counts = np.bincount(loaders['train'].dataset.df.label.to_numpy(dtype=int), minlength=args.classes)
    # A limited training subset can omit a class that still appears in val.
    weights = (counts.sum() / (args.classes * np.maximum(counts, 1))).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    run_dir = RUNS_DIR / f'{args.mode}_{args.fusion}_c{args.classes}{args.tag}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps({**vars(args), 'batch': batch_size, 'seed': SEED}, indent=2))
    history, best, stale = [], -1., 0
    for epoch in range(1, args.epochs + 1):
        stats = {}
        for phase in ('train', 'val'):
            training = phase == 'train'
            model.train(training)
            loss_sum, weight_sum, truth, predictions = 0., 0., [], []
            for images, labels in loaders[phase]:
                images, labels = images.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                        logits = model(images)
                        loss = criterion(logits, labels)
                    if training:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                batch_weight = criterion.weight[labels].sum().item()
                loss_sum += loss.item() * batch_weight
                weight_sum += batch_weight
                truth.extend(labels.cpu().tolist())
                predictions.extend(logits.detach().argmax(1).cpu().tolist())
            stats[f'{phase}_loss'] = loss_sum / weight_sum
            stats[f'{phase}_acc'] = float(np.mean(np.array(truth) == predictions))
            if not training:
                stats['val_f1'] = f1_score(truth, predictions, labels=list(range(args.classes)), average='macro', zero_division=0)
        history.append({'epoch': epoch, **stats})
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        print(f'Epoch {epoch}: {stats}')
        if stats['val_f1'] > best:
            best, stale = stats['val_f1'], 0
            torch.save({'state_dict': model.state_dict(), 'mode': args.mode,
                        'fusion': args.fusion, 'classes': args.classes}, run_dir / 'best.pt')
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['hybrid', 'cnn', 'vit'], default='hybrid')
    parser.add_argument('--fusion', choices=['concat', 'attention', 'none'], default='attention')
    parser.add_argument('--classes', type=int, choices=[2, 3], default=2)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--backbone-lr', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--tag', default='', help='suffix for run name, e.g. _v2')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 15 or args.batch < 1 or (args.limit is not None and args.limit < 1):
        parser.error('epochs must be 1–15; batch and limit must be positive')
    if args.mode != 'hybrid':
        args.fusion = 'none'
    elif args.fusion == 'none':
        parser.error('hybrid requires concat or attention fusion')
    try:
        return run_training(args, args.batch)
    except torch.cuda.OutOfMemoryError:
        if args.batch <= 16:
            raise
    # Outside the except block so the failed training frame can be released.
    torch.cuda.empty_cache()
    print('CUDA out of memory; restarting training with batch size 16')
    return run_training(args, 16)


if __name__ == '__main__':
    main()
