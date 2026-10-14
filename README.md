# PashtoNews-24K Benchmark

Code and experimental records for single-label Pashto news-topic classification using Logistic Regression, linear SVM, a multilayer perceptron (MLP), supervised FastText, and XLM-RoBERTa.

**Author:** Mohammad Anwar Osmani

The cleaned corpus contains **24,122 articles across seven categories**. All five models used the same saved outer training and test files: **19,297 training articles and 4,825 test articles**.

## Repository status

This repository is intended to accompany the associated manuscript.

**Dataset repository:** ?

**Associated manuscript: ?

## Dataset

Articles were collected through September 2026 from accessible category archives of:

- [Pajhwok Afghan News](https://pajhwok.com/ps/)
- [BBC News Pashto](https://www.bbc.com/pashto)
- [Voice of America Pashto](https://www.pashtovoa.com/)
- [Taand](https://taand.net/)
- [TOLOnews Pashto](https://www.tolonews.com/pa)

Source-specific Python scripts traversed category archives using pagination or load-more mechanisms, collected article URLs, and extracted article content. Preparation included cleaning, length checks, and checks for Pashto-specific characters. Initial labels came from targeted publisher categories and were mapped to a common scheme. These are publisher-derived labels, not independently annotated gold-standard labels.


### Categories and shared splits

| Category | Full corpus | Training | Test |
|---|---:|---:|---:|
| economy | 7,520 | 6,016 | 1,504 |
| health | 4,322 | 3,457 | 865 |
| migration | 915 | 732 | 183 |
| politics_security | 5,654 | 4,523 | 1,131 |
| religion_culture | 986 | 789 | 197 |
| science_tech | 1,774 | 1,419 | 355 |
| sports | 2,951 | 2,361 | 590 |
| **Total** | **24,122** | **19,297** | **4,825** |

The outer split used category stratification, test proportion 0.2, and random seed 42. Use the supplied files rather than generating another split when reproducing the reported benchmark.

### CSV fields

| Column | Description |
|---|---|
| `category` | Single topic label |
| `title` | Article headline |
| `text` | Article body used for classification |
| `url` | Source article URL |
| `source` | Publisher or website identifier |
| `body_hash` | SHA-256 of the normalized body |
| `article_id` | Stable identifier for partition checks and prediction alignment |

Only `text` was used as the predictive input; `category` was the target. Other columns were retained as metadata.

### Duplicate handling

Duplicate comparison used body-only normalization: whitespace collapse, trimming, and lowercasing, followed by SHA-256 hashing.

The 25,287 input records formed 24,387 distinct normalized-body groups, including 883 repeated-body groups. All 530 records from 265 conflicting-label groups were excluded. A further 635 redundant copies were removed from label-consistent groups, retaining the first occurrence.

**25,287 − 530 − 635 = 24,122 articles.**

This procedure addresses exact normalized-body duplicates, not all near-duplicates or related stories.

## Repository layout

The following layout is recommended for the accompanying files:

| Path | Contents |
|---|---|
| `classifiers/` | Final LR, SVM, MLP, and FastText scripts |
| `notebooks/` | Final XLM-R Colab notebook |
| `collection/` | Source-specific scraping scripts |
| `preparation/` | Cleaning, deduplication, and split-generation scripts |
| `GeneratingFigures/` | Figure scripts |
| `model_results`| Reported configurations, reports, confusion matrices, and predictions |


## Experimental settings

### TF-IDF classifiers

LR, SVM, and MLP used word unigrams and bigrams, `max_features=10000`, `min_df=5`, and `max_df=0.8`. Their recorded final vocabularies contained 10,000 combined unigram/bigram features. The cap was fixed following an earlier vocabulary-size validation comparison.

The representation used lowercasing, raw counts, smoothed IDF, and L2 normalization, with the default token pattern. No Pashto stopword list, stemming, or lemmatization was added. The vectorizer was fitted on shared training text; test text was transformed using the fitted representation.

| Model | Main settings |
|---|---|
| LR | L-BFGS, L2 regularization, inverse regularization strength `C=1.0`, tolerance `0.0001`, maximum 1,000 optimizer iterations |
| Linear SVM | `LinearSVC`, `C=1.0`, L2 penalty, squared-hinge loss, one-versus-rest, `dual='auto'`, tolerance `0.0001`, maximum 10,000 optimizer iterations |
| MLP | One 100-unit hidden layer, ReLU, Adam, `alpha=0.0001`, initial learning rate `0.001`, automatic minibatches, maximum 500 epochs, internal early stopping |

LR and SVM used the full shared training partition without class weighting or a final regularization-parameter search. Their iteration limits are not directly comparable with neural-network epochs.

MLP reserved 10% internally for accuracy-based early stopping, with tolerance `0.0001` and patience setting `n_iter_no_change=10`. TF-IDF was fitted before the internal split, so validation text contributed to vocabulary and IDF estimation, but not MLP weight updates. The outer test set did not contribute to feature fitting. Labels were encoded for fitting and decoded for evaluation.

### Supervised FastText

FastText used whitespace-consolidated article bodies without lowercasing, formatted as one labeled article per line. Settings were `dim=100`, `wordNgrams=2`, `epoch=25`, `lr=0.5`, `loss='softmax'`, and `thread=4`.

**The reported run disabled character subwords: `minn=0`, `maxn=0`.** Do not enable `--fasttext-subwords` when reproducing that configuration. The script filename does not imply that subwords were enabled.

No pretrained vectors or TF-IDF features were supplied. Training followed a fixed schedule without validation-based checkpoint selection. Seed 42 was recorded as metadata but was not explicitly passed to FastText training; multithreaded execution may vary between runs.

### XLM-RoBERTa

`xlm-roberta-base` was fine-tuned with seven category outputs, maximum input length 256 tokens, truncation, and fixed padding. Seed 42 was set before initialization.

- Fitting articles: 17,367; validation articles: 1,930.
- Four configured training epochs; learning rate `5e-5`.
- Linear learning-rate schedule; 500 warmup steps; weight decay `0.01`.
- Training batch size 16 per device; gradient accumulation 2; nominal effective batch size 32 on one GPU.
- Evaluation batch size 32; FP16 mixed precision; fused PyTorch AdamW.
- Evaluation and checkpoint saving after each epoch.
- Best checkpoint restored according to validation macro-F1.

The saved run selected `checkpoint-2172`, with validation macro-F1 **0.9223**. This internal validation score is distinct from test macro-F1 **0.9309**. Checkpoint selection is not an early-stopping callback. MLP and XLM-R validation subsets should not be assumed to contain identical articles.

Pretrained model revision: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.

## Environment

LR, SVM, MLP, and FastText were trained on the author's personal laptop. XLM-R was fine-tuned on Google Colab's free tier using a T4 GPU and FP16 mixed precision.

| Environment | Recorded versions |
|---|---|
| Laptop | Python 3.12.8; scikit-learn 1.8.0 |
| Colab | Python 3.13.15; PyTorch 2.11.0+cu130; Transformers 4.57.1; Datasets 4.8.5; pandas 2.2.3; scikit-learn 1.6.1 |

Consult each model's configuration for its recorded settings. Add the actual FastText package version and complete dependency specifications before finalizing the reproduction package. Dependency files should describe the environments actually used rather than guessed versions.

Different hardware and software environments mean training times do not establish a controlled computational-efficiency ranking. Fixed seeds do not guarantee bitwise-identical outputs.

## Running the laptop classifiers

Run commands from the repository root. Place the exact downloaded shared CSVs in `data/`.

Core dependencies are pandas, NumPy, scikit-learn, and joblib. FastText additionally requires `fasttext` or a compatible `fasttext-wheel` installation. Use the recorded environment specifications for close reproduction.

### Logistic Regression

```bash
python classifiers/TF-IDFlogisticRegression.py --train data/train_data.csv --test data/test_data.csv --out runs/LR
```

### Linear SVM

```bash
python classifiers/TF-IDFplusSVM.py --train data/train_data.csv --test data/test_data.csv --out runs/SVM
```

### MLP

```bash
python classifiers/MLPClassifier.py --train data/train_data.csv --test data/test_data.csv --out runs/MLP
```

### FastText without character subwords

```bash
python classifiers/FastTextSubwordEmbeddings.py --train data/train_data.csv --test data/test_data.csv --out runs/FastText
```

These scripts load existing partitions. They do not recreate the outer split or deduplicate articles. MLP additionally forms an internal validation subset.

## Running XLM-R in Colab

1. Open the final notebook in `notebooks/` using Google Colab.
2. Select a GPU runtime.
3. Put the exact shared CSV files in a Google Drive folder.
4. Set the notebook's data path to the mounted folder, for example `/content/drive/MyDrive/Pashto_Dataset`.
5. Run the cells in order and retain the output configuration and training history.

Use a mounted filesystem path, not a Google Drive sharing URL. Preserve validation-based checkpoint selection and evaluate on the test partition after selection.

## Reported test results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.9335 | 0.9043 |
| Linear SVM | 0.9389 | 0.9162 |
| MLP | 0.9399 | 0.9182 |
| FastText | 0.9364 | 0.9131 |
| XLM-R | **0.9480** | **0.9309** |

Scores are rounded to four decimal places from saved classification reports. XLM-R had the highest overall accuracy and macro-F1. MLP had the highest Science and Technology F1 (0.9056), while SVM had the highest Sports F1 (0.9847). These comparisons do not establish statistical significance.

## Input integrity and output records

All five saved configurations recorded identical outer input hashes:

```text
train_data.csv SHA-256:
faa106fc35a4122349121549d7073573f07a882204a7acfcce3689fdfaacb7bb

test_data.csv SHA-256:
76038e40d06cb36818e982291c946ec62aaa778a02f8867da7172de15db0c988
```

Hashes refer to exact file bytes; resaving a CSV can change its hash even when its logical contents appear unchanged.

Reported records should include model configurations, classification reports, confusion matrices, and predictions linked to `article_id`. LR also exports the highest positive coefficients to `top_lexical_features.csv`; these are learned predictive associations, not necessarily semantically meaningful topic words.

Do not use the test set to select new settings and then present the resulting scores as an untouched final evaluation.

## Limitations and rights

The corpus is imbalanced and uses publisher-derived labels. Source styles, categorization differences, and near-duplicate stories may affect results. 

News articles may contain sensitive content, personal names, disputed claims, and social or political biases. Inclusion does not imply endorsement.

Underlying news text remains subject to applicable source rights and terms. Repository access does not itself grant permission to redistribute it. 

## Citation and contact

Author: **Mohammad Anwar Osmani**.
