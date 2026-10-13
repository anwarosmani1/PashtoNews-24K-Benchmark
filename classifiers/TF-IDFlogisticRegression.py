"""Standalone Pashto classifier using previously saved shared partitions.
Run: python TF-IDFlogisticRegression.py --train train_data.csv --test test_data.csv
Dependencies: pandas numpy scikit-learn joblib
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
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    parser.add_argument('--out', type=Path, default=Path('results_LR'))
    
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
    
    vectorizer = TfidfVectorizer(max_features=10000, **TFIDF_BASE)    
    start = time.perf_counter()
    x_train = vectorizer.fit_transform(train['text']) 
    x_test = vectorizer.transform(test['text']) 
    feature_seconds = time.perf_counter() - start
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    start = time.perf_counter()
    model.fit(x_train, train['category'])
    training_seconds = time.perf_counter() - start
    joblib.dump(vectorizer, args.out / 'tfidf_vectorizer.joblib')
    joblib.dump(model, args.out / 'LR.joblib')
    
    settings = dict(max_features=10000, actual_vocabulary_size=len(vectorizer.vocabulary_),
                    tfidf=TFIDF_BASE, feature_extraction_seconds=feature_seconds,
                    classifier=model.get_params(),
                    selection_rule='Fixed at 10000 based on previous validation comparison')

    predictions = model.predict(x_test)
    
    features = vectorizer.get_feature_names_out()
    #features = vectorizer.get_feature_names_out() retrieves the names of the features (e.g., words) that the vectorizer has extracted from the input data.
    lexical = []
    for category, weights in zip(model.classes_, model.coef_):
        indices = np.flatnonzero(weights > 0) # finds the indices of the elements in the weights array that are greater than zero.
        
        indices = indices[np.argsort(weights[indices])[::-1]][:5]        
        for rank, index in enumerate(indices, 1):
            lexical.append(dict(category=category, rank=rank, feature=features[index],
                                coefficient=float(weights[index])))
    pd.DataFrame(lexical).to_csv(args.out / 'top_lexical_features.csv',
                               index=False, encoding='utf-8-sig')

    report = classification_report(test['category'], predictions, labels=classes,
                                   output_dict=True, zero_division=0)
    printable = classification_report(test['category'], predictions, labels=classes,
                                      digits=4, zero_division=0)
    print(printable)
    print(f"Accuracy: {accuracy_score(test['category'], predictions):.4f}")
    print(f"Macro-F1: {report['macro avg']['f1-score']:.4f}")
    (args.out / 'LR_report.txt').write_text(printable, encoding='utf-8')
    (args.out / 'LR_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(confusion_matrix(test['category'], predictions, labels=classes),
                 index=classes, columns=classes).to_csv(args.out / 'LR_confusion.csv')
    predictions_table = test[['article_id', 'category']].rename(
        columns={'category': 'true_label'}).copy()
    predictions_table['LR'] = predictions
    predictions_table.to_csv(args.out / 'test_predictions.csv', index=False,
                             encoding='utf-8-sig')
    summary = dict(model='LR', accuracy=accuracy_score(test['category'], predictions),
                   macro_precision=report['macro avg']['precision'],
                   macro_recall=report['macro avg']['recall'],
                   macro_f1=report['macro avg']['f1-score'],
                   weighted_f1=report['weighted avg']['f1-score'],
                   training_seconds=training_seconds)
    pd.DataFrame([summary]).to_csv(args.out / 'model_comparison.csv', index=False)
    metadata = dict(model='LR', seed=SEED,
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
