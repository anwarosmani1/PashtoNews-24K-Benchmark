"""Create shared stratified train/test CSVs without training any model.
Run button: put Pashto-news-24K.jsonl beside this script.
CLI: python prepare_train_test.py --data PATH --out FOLDER
Dependencies: pandas numpy scikit-learn.
Input is already cleaned: no deduplication or text normalization is performed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def load_data(path):
    rows = []
    with path.open(encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Line {line_number}: invalid JSON') from error
            if not isinstance(row, dict):
                raise ValueError(f'Line {line_number}: expected a JSON object')
            for field in ('text', 'category', 'body_hash'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'Line {line_number}: invalid {field}')
            # Reuse the identifier from your completed cleanup.
            row['article_id'] = row['body_hash']
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty or df['category'].nunique() < 3:
        raise ValueError('Expected a nonempty dataset with at least three classes')
    return df


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=script_dir / 'Pashto-news-24K.jsonl')
    parser.add_argument('--out', type=Path, default=script_dir)
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f'Dataset not found: {args.data.resolve()}. '
            'Put articles_final_cleaned.jsonl beside this script or supply --data.')
    outputs = [args.out / name for name in
               ('train_data.csv', 'test_data.csv', 'split_summary.json')]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Already exists: {path}. Reuse existing splits or choose another --out folder.')
    df = load_data(args.data)
    train_ids, test_ids = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED,
        stratify=df['category'])
    train, test = df.iloc[train_ids].copy(), df.iloc[test_ids].copy()
    summary = dict(
        input_file=str(args.data.resolve()),
        dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        seed=SEED, test_size=0.2, stratified_by='category',
        input_records=len(df), training_records=len(train), test_records=len(test),
        training_category_counts={k:int(v) for k,v in train['category'].value_counts().sort_index().items()},
        test_category_counts={k:int(v) for k,v in test['category'].value_counts().sort_index().items()},
        note='Input order preserved before splitting; text and categories are unchanged.')
    args.out.mkdir(parents=True, exist_ok=True)
    for path, subset in zip(outputs[:2], (train, test)):
        with path.open('x', encoding='utf-8-sig', newline='') as handle:
            subset.to_csv(handle, index=False)
    with outputs[2].open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f'Articles: {len(df)}; train: {len(train)}; test: {len(test)}')
    for path in outputs:
        print(f'Saved: {path.resolve()}')


if __name__ == '__main__':
    main()
