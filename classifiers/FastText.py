"""Standalone Pashto classifier using previously saved shared partitions.
Run: python FastTextSubwordEmbeddings.py --train train_data.csv --test test_data.csv
Dependencies: pandas numpy scikit-learn fasttext (or fasttext-wheel)
No dataset splitting or deduplication is performed here.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


SEED = 42
TFIDF_BASE = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)


def load_partition(path):
    df = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
    if df.empty:
        raise ValueError(f'Empty partition: {path}')
    for field in ('text', 'category'):
        if field not in df or not df[field].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f'{path}: {field} must contain nonempty strings')
    if 'article_id' not in df:
        if 'body_hash' not in df:
            raise ValueError(f'{path}: expected article_id or body_hash')
        df['article_id'] = df['body_hash']
    if not df['article_id'].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(f'{path}: invalid article identifiers')
    return df


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('train_data.csv'))
    parser.add_argument('--test', type=Path, default=Path('test_data.csv'))
    parser.add_argument('--out', type=Path, default=Path('results_FastText'))
    parser.add_argument('--fasttext-subwords', action='store_true',
                        help='Enable character n-grams of lengths 3 through 6')
    args = parser.parse_args()
    train, test = load_partition(args.train), load_partition(args.test)
    # Check saved partition separation; this does not deduplicate any articles.
    if set(train['article_id']) & set(test['article_id']):
        raise ValueError('Training and test partitions overlap by article_id')
    classes = sorted(train['category'].unique())
    if len(classes) < 3 or not set(test['category']).issubset(classes):
        raise ValueError('Expected at least three training categories and known test labels')
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Train: {len(train)}; test: {len(test)}')
    try:
        import fasttext
    except ImportError as error:
        raise ImportError('Install fasttext or fasttext-wheel on Windows') from error
    def clean(text):
        return ' '.join(text.split())
    training_file = args.out / 'fasttext_train.txt'
    with training_file.open('w', encoding='utf-8') as handle:
        for row in train.itertuples():
            if any(char.isspace() for char in row.category):
                raise ValueError('FastText category identifiers cannot contain whitespace')
            handle.write(f'__label__{row.category} {clean(row.text)}\n')
    config = dict(lr=0.5, epoch=25, wordNgrams=2, dim=100, loss='softmax',
                  thread=4, minn=3 if args.fasttext_subwords else 0,
                  maxn=6 if args.fasttext_subwords else 0)
    start = time.perf_counter()
    model = fasttext.train_supervised(input=str(training_file), **config)
    training_seconds = time.perf_counter() - start
    predictions = [model.predict(clean(text), k=1)[0][0].removeprefix('__label__')
                   for text in test['text']]
    model.save_model(str(args.out / 'fasttext.bin'))
    settings = dict(config=config, preprocessing='Collapse whitespace; no lowercasing',
                    note='TF-IDF settings and SEED do not configure FastText; multithreaded training may vary')
    (args.out / 'fasttext_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    report = classification_report(test['category'], predictions, labels=classes,
                                   output_dict=True, zero_division=0)
    printable = classification_report(test['category'], predictions, labels=classes,
                                      digits=4, zero_division=0)
    print(printable)
    print(f"Accuracy: {accuracy_score(test['category'], predictions):.4f}")
    print(f"Macro-F1: {report['macro avg']['f1-score']:.4f}")
    (args.out / 'FastText_report.txt').write_text(printable, encoding='utf-8')
    (args.out / 'FastText_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(confusion_matrix(test['category'], predictions, labels=classes),
                 index=classes, columns=classes).to_csv(args.out / 'FastText_confusion.csv')
    predictions_table = test[['article_id', 'category']].rename(
        columns={'category': 'true_label'}).copy()
    predictions_table['FastText'] = predictions
    predictions_table.to_csv(args.out / 'test_predictions.csv', index=False,
                             encoding='utf-8-sig')
    summary = dict(model='FastText', accuracy=accuracy_score(test['category'], predictions),
                   macro_precision=report['macro avg']['precision'],
                   macro_recall=report['macro avg']['recall'],
                   macro_f1=report['macro avg']['f1-score'],
                   weighted_f1=report['weighted avg']['f1-score'],
                   training_seconds=training_seconds)
    pd.DataFrame([summary]).to_csv(args.out / 'model_comparison.csv', index=False)
    metadata = dict(model='FastText', seed=SEED,
                    train_file_sha256=hashlib.sha256(args.train.read_bytes()).hexdigest(),
                    test_file_sha256=hashlib.sha256(args.test.read_bytes()).hexdigest(),
                    training_records=len(train), test_records=len(test),
                    python=platform.python_version(), sklearn=sklearn.__version__,
                    settings=settings)
    (args.out / 'experiment_config.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Outputs: {args.out.resolve()}')


if __name__ == '__main__':
    main()
