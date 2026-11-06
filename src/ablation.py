"""Summarize all evaluated runs in a CSV and Markdown table."""
import argparse
import json
import pandas as pd
from src.config import OUT_DIR, RUNS_DIR
from src.models import build_model


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    rows = []
    for path in sorted(RUNS_DIR.glob('*/metrics.json')):
        metrics = json.loads(path.read_text())
        config = json.loads((path.parent / 'config.json').read_text())
        model = build_model(config['mode'], config['fusion'], config['classes'], pretrained=False)
        rows.append(dict(run=path.parent.name, params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                         accuracy=metrics['accuracy'], precision=metrics['precision_macro'],
                         recall=metrics['recall_macro'], f1=metrics['f1_macro'], roc_auc=metrics['roc_auc']))
    df = pd.DataFrame(rows, columns=['run', 'params_M', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'ablation.csv', index=False)
    # Avoid an optional tabulate dependency for a simple report table.
    lines = ['| ' + ' | '.join(df.columns) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |']
    for row in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    (OUT_DIR / 'ablation.md').write_text('\n'.join(lines) + '\n')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
