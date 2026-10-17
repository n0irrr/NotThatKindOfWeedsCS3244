# Iteration Log

One row per experiment. Change **one factor at a time** so improvements are attributable.

Rows 0–4 used split v1 (sklearn `StratifiedGroupKFold`), where all of val's Parkinsonia
and all of test's Siam weed came from a single capture session each, and the seed had
no effect. Rows 5+ use split v2 (`src/data/split.py`): no session is more than 25% of
a class's val/test images, and every class spans 16+ sessions in val and in test.
Compare new models against rows 5+; rows 3–4 need re-running on split v2.

| # | Date | Model | What changed | Macro F1 | Macro P | Macro R | Acc | Notes / hypothesis for next step |
|---|------|-------|--------------|----------|---------|---------|-----|----------------------------------|
| 0 | 2026-10-15 | Majority class | Baseline (always predict Negative) | 0.076 | 0.058 | 0.111 | 0.522 | Accuracy looks fine, macro F1 shows it's useless |
| 1 | 2026-10-15 | KNN (k=7, uniform) | Flattened 32x32 pixels, scaled + PCA(100) | 0.166 | 0.219 | 0.194 | 0.461 | Barely beats majority; Rubber vine F1=0. Pixel distance tracks lighting/background, not species. Try colour features |
| 2 | 2026-10-15 | KNN (k=11, uniform) | HSV colour histograms (3x32 bins) instead of pixels | 0.499 | 0.548 | 0.473 | 0.660 | Big jump over raw pixels. Main error: weeds -> Negative. Val F1 0.398 vs test 0.499, so split variance is high. Next: CNN |
| 3 | 2026-10-17 | Simple CNN (4 blocks, 1.2M params) | Learned features from scratch, 128x128, Adam 1e-3 + cosine, 30 epochs, no aug/class weights; best = epoch 27 (resumed from epoch 10 weights with fresh Adam state) | 0.783 | 0.815 | 0.762 | 0.843 | +0.28 over KNN-hist. Gains came late as the LR decayed (epoch 10 test F1 was 0.625). Train acc 0.98 vs val F1 0.67, so heavy overfitting. Main errors: Siam weed / Parkinsonia / Snake weed -> Negative, Snake weed -> Chinee apple. Next: augmentation |
| 4 | 2026-10-17 | ResNet50 (ImageNet, 23.5M params) | Pretrained backbone instead of from-scratch CNN; same 128x128, no aug/class weights. 2 head-only epochs (lr 1e-3) then full fine-tune (lr 1e-4, cosine); early-stopped at 14, best = epoch 9 | 0.787 | 0.795 | 0.783 | 0.841 | Test F1 ~ simple CNN (0.783) but val F1 clearly higher (0.721 vs 0.673), reached in 9 epochs / 23 min vs 27 epochs. Train acc 0.99+ by epoch 7, so overfits hard. Weakest: Snake weed, Chinee apple. Next: augmentation, then 224x224 |
| 5 | 2026-10-17 | Majority class | Split v2 (only change vs #0) | 0.077 | 0.058 | 0.111 | 0.525 | Floor for split v2 rows |
| 6 | 2026-10-17 | KNN (k=3, uniform) | Split v2 (only change vs #1) | 0.180 | 0.218 | 0.192 | 0.441 | Val 0.175 vs test 0.180, so the val/test gap is gone. Still tracks lighting: Negative -> Parkinsonia 264 |
| 7 | 2026-10-17 | KNN (k=11, distance) | Split v2 (only change vs #2) | 0.453 | 0.493 | 0.437 | 0.581 | Test 0.499 -> 0.453 but val/test gap halves (0.406 vs 0.453): #2's test score was helped by a one-session Siam weed test set. Baseline for the CNN |