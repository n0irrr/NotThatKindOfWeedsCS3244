# Iteration Log

One row per experiment. Change **one factor at a time** so improvements are attributable.

Rows 0–2 used split v1 (sklearn `StratifiedGroupKFold`), where all of val's Parkinsonia
and all of test's Siam weed came from a single capture session each, and the seed had
no effect. Rows 3+ use split v2 (`src/data/split.py`): no session is more than 25% of
a class's val/test images, and every class spans 16+ sessions in val and in test.
Compare new models against rows 3+.

| # | Date | Model | What changed | Macro F1 | Macro P | Macro R | Acc | Notes / hypothesis for next step |
|---|------|-------|--------------|----------|---------|---------|-----|----------------------------------|
| 0 | 2026-10-15 | Majority class | Baseline (always predict Negative) | 0.076 | 0.058 | 0.111 | 0.522 | Accuracy looks fine, macro F1 shows it's useless |
| 1 | 2026-10-15 | KNN (k=7, uniform) | Flattened 32x32 pixels, scaled + PCA(100) | 0.166 | 0.219 | 0.194 | 0.461 | Barely beats majority; Rubber vine F1=0. Pixel distance tracks lighting/background, not species. Try colour features |
| 2 | 2026-10-15 | KNN (k=11, uniform) | HSV colour histograms (3x32 bins) instead of pixels | 0.499 | 0.548 | 0.473 | 0.660 | Big jump over raw pixels. Main error: weeds -> Negative. Val F1 0.398 vs test 0.499, so split variance is high. Next: CNN |
| 3 | 2026-10-17 | Majority class | Split v2 (only change vs #0) | 0.077 | 0.058 | 0.111 | 0.525 | Floor for rows 4+ |
| 4 | 2026-10-17 | KNN (k=3, uniform) | Split v2 (only change vs #1) | 0.180 | 0.218 | 0.192 | 0.441 | Val 0.175 vs test 0.180, so the val/test gap is gone. Still tracks lighting: Negative -> Parkinsonia 264 |
| 5 | 2026-10-17 | KNN (k=11, distance) | Split v2 (only change vs #2) | 0.453 | 0.493 | 0.437 | 0.581 | Test 0.499 -> 0.453 but val/test gap halves (0.406 vs 0.453): #2's test score was helped by a one-session Siam weed test set. Baseline for the CNN |
