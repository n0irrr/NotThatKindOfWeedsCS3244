# Not that kind of Weed

CS3244 project: automated weed species classification on the
[DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) dataset — 17,509 in-situ RGB field images across 8 Australian weed species + a Negative class, collected at 8 Queensland sites.

## Key questions

1. What simple classical ML approaches work as a first baseline?
2. How much does a CNN that learns features directly improve over classical methods?
3. How do tuning, preprocessing, augmentation and transfer learning improve performance while limiting overfitting?
4. Which species remain confused by the best model, and why?

## Pipeline

```
data/                 raw images + labels (gitignored), processed arrays, split CSVs
notebooks/            01_eda → 02_baseline → 03_cnn → 04_transfer → 05_error_analysis
src/
  data/               split.py (session-grouped stratified 60/20/20; random split for comparison)
                      dataset.py (in-memory arrays for classical models, lazy Dataset for CNNs)
                      stats.py (train-only normalisation stats)
  features/           classical.py (pixels, histograms) · augment.py (CNN transforms + augmentations)
  models/             knn.py · simple_cnn.py · transfer.py
  evaluation/         metrics.py (macro P/R/F1) · confusion.py
  train.py            entry point, driven by a YAML config
experiments/
  configs/            one YAML per experiment
  ITERATION_LOG.md    one-factor-at-a-time experiment log
results/              figures, confusion matrices, metrics
reports/              proposal, final report
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data (gitignored): put the 17,509 JPGs in `data/images/` and the official `labels.csv`
in `data/`. The authors' Google Drive link is often over quota; the Kaggle mirror
(`kaggle.com/datasets/imsparsh/deepweeds`, images in `images/`) has the same files.
`data/images` can also be a link to a folder elsewhere.

```bash
curl -L -o data/labels.csv https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv
python -m src.data.split               # session-grouped split -> data/splits/ (committed)
python -m src.data.split --no-group    # random split -> data/splits_random/ (committed)
python -m src.data.stats               # train-only normalisation stats (committed)
python -m src.train --config experiments/configs/knn_hist.yaml
```

Splits and stats are already in the repo, so only re-run those steps if you
change the split logic or `data.image_size`.

## Evaluation

Headline metric is **macro F1** (Negative class is ~52% of data, so accuracy alone
is misleading), with macro precision/recall, per-class breakdown and a confusion
matrix for every run. Anchored against the majority-class baseline and the
published 95.7% ResNet50 result. Stratified splits + fixed seeds throughout.
