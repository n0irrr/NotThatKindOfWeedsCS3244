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
  data/               kaggle_split.py (the authors' 60/20/20 split from the Kaggle zip, fold 0 -> data/splits_kaggle0/)
                      dataset.py (in-memory arrays for classical models, lazy Dataset for CNNs)
                      stats.py (train-only normalisation stats)
  features/           classical.py (pixels, colour histograms, HOG) · augment.py (CNN transforms + augmentations)
  models/             knn.py · logreg.py · simple_cnn.py · transfer.py
  evaluation/         metrics.py (macro P/R/F1) · confusion.py · aggregate.py (mean ± std over seeds)
                      explain.py (Grad-CAM, LIME, error gallery)
                      tta.py (flip / rotate test-time augmentation, seed ensembles)
  train.py            entry point, driven by a YAML config
experiments/
  configs/            one YAML per experiment
  logs/               console log of every run
  ITERATION_LOG.md    one-factor-at-a-time experiment log
results/              figures, confusion matrices, metrics, per-image predictions
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
python -m src.data.kaggle_split path/to/kaggle/labels --fold 0   # -> data/splits_kaggle0/ (committed)
python -m src.train --config experiments/configs/knn_hist_kaggle0.yaml
```

The split and train-only stats are already in the repo (stats are created on
first use at each image size), so you don't need to re-run those steps.
`labels/` in the Kaggle zip holds the authors' 5 folds (`{train,val,test}_subset{0-4}.csv`,
identical to the copies on the authors' GitHub); we use fold 0.

## Running experiments

```bash
# tuning run: scores val only, never loads test
python -m src.train --config experiments/configs/resnet50_aug_geo_crop_30ep_bf16_kaggle0.yaml --seed 1 --val-only
# once a config is chosen on val: score its checkpoint on test
python -m src.train --config experiments/configs/resnet50_aug_geo_crop_30ep_bf16_kaggle0.yaml --seed 1 --eval-only
# repeat runs with other seeds, then mean ± std and a pooled confusion matrix
python -m src.train --config experiments/configs/resnet50_aug_geo_crop_30ep_bf16_kaggle0.yaml --seed 2
python -m src.evaluation.aggregate resnet50_aug_geo_crop_30ep_bf16_kaggle0
# test-time augmentation (8 flip / rotate views) and the seed ensemble; scores val unless --split test
python -m src.evaluation.tta experiments/configs/resnet50_aug_geo_crop_30ep_bf16_kaggle0.yaml --seeds 1 2 3
```

Every config reads the authors' split, fold 0 (`data/splits_kaggle0/`, hence the `_kaggle0` suffix).

`--resume` continues an interrupted CNN run from its last epoch. On an Intel
GPU the `*_bf16.yaml` configs (bfloat16 + channels_last) run ~40% faster.

## Leakage rules

1. **No image in more than one split.** All results use the authors' split as
   published on Kaggle, fold 0 (10,501 train / 3,501 val / 3,507 test).
   `tests/test_kaggle_split.py` checks that no image is in two splits, every
   image is used once and labels match `labels.csv`. The authors' folds split
   by image, not by capture session, so frames taken seconds apart in the same
   session can be in train and in test.
2. **Normalisation stats from train only**; augmentation on train only.
3. **Test never drives a choice**: k / C and the best epoch are picked on val,
   and each CNN's augmentation was chosen on validation data in earlier tuning
   runs, before the final runs. Tuning runs use `--val-only`. Test-time
   augmentation and the seed ensemble were also chosen on val
   (`experiments/ITERATION_LOG.md` rows 12–13).
4. **Pretrained weights must predate DeepWeeds**: ImageNet (2012) is fine.
   Web-scale models such as CLIP are not used, since their training data may
   include DeepWeeds images.

## Evaluation

Headline metric is **macro F1** (Negative class is ~52% of data, so accuracy alone
is misleading), with macro precision/recall, per-class breakdown and a confusion
matrix for every run. Final configs are repeated over seeds and reported as
mean ± std. Anchored against the majority-class baseline. For reference, the
paper reports 95.7% accuracy for ResNet50 averaged over all 5 folds at 224×224;
ours uses fold 0 only, at 128×128 and (rows 15–16) 224×224.

## Results (Kaggle split, fold 0, test set)

| Model | Macro F1 | Macro precision | Macro recall | Accuracy |
|---|---|---|---|---|
| Majority class (always Negative) | 0.076 | 0.058 | 0.111 | 0.520 |
| KNN, raw pixels | 0.192 | 0.204 | 0.207 | 0.434 |
| KNN, HOG | 0.128 | 0.149 | 0.131 | 0.388 |
| KNN, HSV colour histogram (best classical) | 0.613 | 0.650 | 0.586 | 0.721 |
| Logistic regression, raw pixels | 0.234 | 0.268 | 0.220 | 0.504 |
| Logistic regression, HOG | 0.166 | 0.185 | 0.162 | 0.434 |
| Logistic regression, HSV colour histogram | 0.375 | 0.464 | 0.341 | 0.595 |
| Simple CNN, no augmentation (1 seed) | 0.870 | 0.875 | 0.865 | 0.900 |
| Simple CNN + flip / rotate / colour jitter (3 seeds) | 0.905 ± 0.005 | 0.915 ± 0.004 | 0.895 ± 0.006 | 0.925 ± 0.004 |
| ResNet-50, no augmentation (1 seed) | 0.870 | 0.878 | 0.865 | 0.900 |
| ResNet-50 + flip / rotate / random crop (3 seeds) | 0.923 ± 0.005 | 0.936 ± 0.006 | 0.910 ± 0.004 | 0.940 ± 0.004 |
| ResNet-50 + flip / rotate / random crop, with TTA (3 seeds) | 0.938 ± 0.004 | 0.951 ± 0.002 | 0.926 ± 0.008 | 0.951 ± 0.004 |
| ResNet-50, ensemble of the 3 seeds, with TTA | 0.948 | 0.958 | 0.939 | 0.959 |
| ResNet-50 + flip / rotate / random crop, 224×224 (3 seeds) | 0.965 ± 0.002 | 0.969 ± 0.001 | 0.962 ± 0.002 | 0.973 ± 0.002 |
| ResNet-50, 224×224, with TTA (3 seeds) | 0.969 ± 0.002 | 0.972 ± 0.001 | 0.966 ± 0.003 | 0.976 ± 0.001 |
| **ResNet-50, 224×224, ensemble of the 3 seeds, with TTA** | **0.973** | **0.976** | **0.970** | **0.978** |

Augmentation adds +0.05 macro F1 for ResNet-50 and +0.035 for the simple CNN, both
well above the seed spread (± 0.005). Test-time augmentation (TTA: averaging each
image's predictions over its 8 flip / 90° rotation views) adds another +0.016, and
averaging the 3 seeds another +0.010, with no retraining. Both were chosen on val
(`experiments/ITERATION_LOG.md` rows 12–13) before test was scored.

Training at 224×224 instead of 128×128 (the only change; row 15) adds +0.042 per run,
and with TTA and the 3-seed ensemble gives the best result, 0.973 (row 16). At 224 px
the gain over a single model comes from the ensemble: on test the ensemble without TTA
also scores 0.973. 256×256, the native size, was worse on val (0.954, seed 1; row 17)
and was not scored on test. A per-class decision offset tuned on val to counter the
pull towards Negative did not generalise (+0.0003 cross-validated within val, row 14)
and was not applied. Details in `notebooks/04_transfer.ipynb` section 7.

Per-class scores, pooled confusion matrices, Grad-CAM, LIME and the error gallery
are in `notebooks/05_error_analysis.ipynb`; figures in
`results/figures/error_analysis_kaggle0/`. Its section 5 shows that every test image
but 3 has a training frame from the same capture session taken within 10 minutes
(the authors' folds split by image, not session), so these scores likely overstate
accuracy on a new field.
