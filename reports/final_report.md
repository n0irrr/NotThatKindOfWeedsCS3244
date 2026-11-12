# Not that kind of Weed: final report

## 1. Problem and data
DeepWeeds: 17,509 images, 8 weed species + Negative, 8 Queensland sites. Session-grouped stratified 60/20/20 split (no capture session straddles train/val/test).

## 2. Methods
KNN baselines, 4-block simple CNN (128x128), ImageNet ResNet50 (128x128). Factors changed one at a time (see `experiments/ITERATION_LOG.md`): geometric augmentation, schedule length, colour augmentation.

## 3. Results (mean ± std over 5 seeds)
Seeds 0-4, session-grouped split, 128x128 input. Source: `results/metrics/multiseed_summary.json` (`python -m src.evaluation.aggregate ...`).

| Run | Val macro F1 | Test macro F1 |
|---|---|---|
| Simple CNN, flip+rotate, 60 ep | 0.692 ± 0.008 | 0.761 ± 0.010 |
| Simple CNN, flip+rotate+colour, 60 ep | 0.681 ± 0.003 | 0.752 ± 0.009 |
| ResNet50, flip+rotate, 30 ep | 0.731 ± 0.007 | 0.787 ± 0.006 |
| ResNet50, flip+rotate+colour, 30 ep | 0.729 ± 0.006 | 0.782 ± 0.006 |

- **CNN vs ResNet50 (geometric augmentation only).** The single-seed tie (0.775 vs 0.774) does not hold up over seeds: ResNet50 leads by ~0.026 test macro F1 (Welch p = 0.002) and ~0.039 val macro F1 (p < 0.001).
- **Colour augmentation does not help.** Simple CNN: val -0.011 (p = 0.035), test -0.009 (p = 0.19). ResNet50: val -0.002 (p = 0.60), test -0.005 (p = 0.30). Only the CNN validation drop is significant at 0.05; with 5 seeds the test differences are within noise, so the honest reading is "no gain", not "harmful".
- **Per class (test F1).** Siam weed (0.90-0.94), Parkinsonia (~0.87) and Rubber vine (0.83-0.87) are easiest. Parthenium is weakest for every configuration (0.445-0.507), then Snake weed (0.62-0.73) and Chinee apple (~0.68-0.69). ResNet50's gains over the CNN are in Lantana, Snake weed, Prickly acacia and Siam weed; for Parthenium it is slightly worse than the CNN (0.467 vs 0.507, both with overlapping spread), and colour augmentation lowers Parthenium for both models.

## 4. Error analysis
Seed 0 of the two geometric-augmentation models on the test set (`python -m src.evaluation.error_analysis`, outputs in `results/error_analysis/`; notebook `notebooks/05_error_analysis.ipynb`). These are single-seed observations, not averaged.

| Class | ResNet50 wrong | Main confusions (ResNet50) | Simple CNN wrong | Main confusions (CNN) |
|---|---|---|---|---|
| Parthenium | 78/173 | Prickly acacia 29, Negative 26, Snake weed 8 | 89/173 | Prickly acacia 48, Negative 24 |
| Snake weed | 29/203 | Chinee apple 9, Negative 7, Siam weed 6 | 48/203 | Chinee apple 21, Negative 13 |
| Chinee apple | 49/225 | Snake weed 18, Negative 14, Lantana 8 | 57/225 | Snake weed 33, Negative 15 |

Overall seed-0 test accuracy: ResNet50 0.824 (612 errors), simple CNN 0.821 (622 errors).

- **Confusions are structured.** Parthenium is mostly mistaken for Prickly acacia and Negative; Snake weed and Chinee apple are confused with each other and with Negative. ResNet50 mainly reduces the Prickly acacia confusion for Parthenium (29 vs 48).
- **Errors concentrate in a few capture sessions.** Session 882 alone has 285 test images with 213 errors (75% error rate) for ResNet50, about 35% of all 612 errors. Several small sessions (506, 512, 513, 528) are 75-100% wrong. For Chinee apple the 3 worst sessions hold 69% (ResNet50) and 79% (CNN) of that class's errors; for Parthenium, a single day (20170906) has a 72-80% error rate. This points to site/lighting/day shift between sessions, which the session-grouped split exposes, more than to the model failing uniformly.
- **Grad-CAM.** Maps for Parthenium, Snake weed and Chinee apple were generated for both models (`*_gradcam_*.png`) but have not been reviewed or interpreted for this report.

## 5. Final evaluation
Selection used mean validation macro F1 only (test hidden during ranking, `python -m src.evaluation.final_eval`):

1. ResNet50, flip+rotate, 30 ep: 0.731 ± 0.007 (selected)
2. ResNet50, flip+rotate+colour, 30 ep: 0.729 ± 0.006
3. Simple CNN, flip+rotate, 60 ep: 0.692 ± 0.008
4. Simple CNN, flip+rotate+colour, 60 ep: 0.681 ± 0.003

**Selected: `resnet50_aug_geo_30ep`. Test macro F1 0.787 ± 0.006, test accuracy 0.815** (5 seeds). Published ResNet50 accuracy is 0.957, so our accuracy is 0.142 lower. Saved in `results/metrics/final_evaluation.json`. Note the top two ResNet50 configurations are within one std on validation, so the choice between them is not decisive.

### Why we expect to sit below 95.7%
- **Stricter split.** Session-grouped: the model never sees images from the same capture session in training, while the published protocol (random 5-fold) puts near-duplicate consecutive frames on both sides. Our own `data/splits_random` split quantifies this (`random_split.yaml`).
- **Smaller input.** 128x128 vs the paper's 256x256 (fine leaf texture is lost; Parthenium / Snake weed / Chinee apple differ in fine detail).
- **Metric.** Paper reports accuracy; we headline macro F1, which is lower than accuracy because Negative is ~52% of data.
- **Budget.** Fewer epochs/seeds, no hyper-parameter search.
_Verify each claim with an experiment before asserting it (random-split run, 224x224 run)._
