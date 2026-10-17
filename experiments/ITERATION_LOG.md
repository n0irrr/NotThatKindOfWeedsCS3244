# Iteration Log

One row per experiment. Change **one factor at a time** so improvements are attributable.

| # | Date | Model | What changed | Macro F1 | Macro P | Macro R | Acc | Notes / hypothesis for next step |
|---|------|-------|--------------|----------|---------|---------|-----|----------------------------------|
| 0 | 2026-10-15 | Majority class | Baseline (always predict Negative) | 0.076 | 0.058 | 0.111 | 0.522 | Accuracy looks fine, macro F1 shows it's useless |
| 1 | 2026-10-15 | KNN (k=7, uniform) | Flattened 32x32 pixels, scaled + PCA(100) | 0.166 | 0.219 | 0.194 | 0.461 | Barely beats majority; Rubber vine F1=0. Pixel distance tracks lighting/background, not species. Try colour features |
| 2 | 2026-10-15 | KNN (k=11, uniform) | HSV colour histograms (3x32 bins) instead of pixels | 0.499 | 0.548 | 0.473 | 0.660 | Big jump over raw pixels. Main error: weeds -> Negative. Val F1 0.398 vs test 0.499, so split variance is high. Next: CNN |
| 3 | 2026-10-17 | Simple CNN (4 blocks, 1.2M params) | Learned features from scratch, 128x128, Adam 1e-3 + cosine, 30 epochs, no aug/class weights; best = epoch 27 (resumed from epoch 10 weights with fresh Adam state) | 0.783 | 0.815 | 0.762 | 0.843 | +0.28 over KNN-hist. Gains came late as the LR decayed (epoch 10 test F1 was 0.625). Train acc 0.98 vs val F1 0.67, so heavy overfitting. Main errors: Siam weed / Parkinsonia / Snake weed -> Negative, Snake weed -> Chinee apple. Next: augmentation |
| 4 | 2026-10-17 | ResNet50 (ImageNet, 23.5M params) | Pretrained backbone instead of from-scratch CNN; same 128x128, no aug/class weights. 2 head-only epochs (lr 1e-3) then full fine-tune (lr 1e-4, cosine); early-stopped at 14, best = epoch 9 | 0.787 | 0.795 | 0.783 | 0.841 | Test F1 ~ simple CNN (0.783) but val F1 clearly higher (0.721 vs 0.673), reached in 9 epochs / 23 min vs 27 epochs. Train acc 0.99+ by epoch 7, so overfits hard. Weakest: Snake weed, Chinee apple. Next: augmentation, then 224x224 |
