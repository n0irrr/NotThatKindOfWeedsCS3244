"""Per-class decision bias for a trained CNN run (ITERATION_LOG row 14).

Most remaining errors of the ResNet-50 ensemble are weeds predicted as Negative, the
majority class. Instead of retraining, add a constant b_c to each class's log-probability
before the argmax, pick b on val to maximise macro F1, and apply the same b to test.
Negative's offset is fixed at 0 (adding the same constant to every class changes nothing),
so 8 numbers are fitted.

    python -m src.evaluation.bias experiments/configs/resnet50_aug_geo_crop_30ep_bf16_kaggle0.yaml --seeds 1 2 3

scores the ensemble + TTA probabilities on val (cached in experiments/cache/), reports a
5-fold cross-validated estimate of the gain within val, fits b on all of val and writes
results/metrics/<run_name>_bias_val.json. Only if the cross-validated gain holds up:

    python -m src.evaluation.bias <same config> --seeds 1 2 3 --split test

applies the saved val offsets to test once; it never fits anything on test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import compute_metrics
from src.evaluation.tta import KEYS, ROOT, combos

GRID = np.round(np.arange(-3.0, 3.0001, 0.1), 2)
EPS = 1e-12


def macro_f1(y, pred, n_classes):
    return f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_bias(log_p, bias):
    return (log_p + bias).argmax(1)


def fit_bias(log_p, y, fixed=None, grid=GRID, rounds=10):
    """Coordinate ascent on macro F1. log_p: (n, C) log-probabilities. `fixed` is the class
    whose offset stays 0. Ties keep the offset closest to 0 so it does not drift."""
    n_classes = log_p.shape[1]
    fixed = n_classes - 1 if fixed is None else fixed
    bias = np.zeros(n_classes)
    best = macro_f1(y, apply_bias(log_p, bias), n_classes)
    order = np.argsort(np.abs(grid), kind="stable")   # 0 first, then +/-0.1, ...
    for _ in range(rounds):
        changed = False
        for c in range(n_classes):
            if c == fixed:
                continue
            keep = bias[c]
            for v in grid[order]:
                bias[c] = v
                score = macro_f1(y, apply_bias(log_p, bias), n_classes)
                if score > best + 1e-9:
                    best, keep, changed = score, v, True
            bias[c] = keep
        if not changed:
            break
    return bias, best


def cv_gain(log_p, y, fixed=None, folds=5, seed=0):
    """Out-of-fold predictions: fit the offsets on 4/5 of the split, predict the other 1/5.
    Returns (macro F1 without offsets, out-of-fold macro F1 with offsets)."""
    n_classes = log_p.shape[1]
    pred = np.empty_like(y)
    for fit_idx, held_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(log_p, y):
        bias, _ = fit_bias(log_p[fit_idx], y[fit_idx], fixed)
        pred[held_idx] = apply_bias(log_p[held_idx], bias)
    return macro_f1(y, log_p.argmax(1), n_classes), macro_f1(y, pred, n_classes)


def ensemble_probs(config_path, seeds, split, root=ROOT):
    """Ensemble + TTA probabilities (n, C) and labels, cached per run / split / seeds."""
    from src.train import load_config

    run_name = load_config(config_path).get("run_name", Path(config_path).stem)
    cache = Path(root) / "experiments" / "cache" / f"{run_name}_{split}_s{''.join(map(str, seeds))}.npz"
    if cache.exists():
        z = np.load(cache)
        return run_name, z["y"], z["p"]
    from src.evaluation.tta import score_views

    _, y, probs_by_seed = score_views(config_path, seeds, split, root=root)
    p = combos(probs_by_seed)["ensemble + TTA" if len(seeds) > 1 else f"s{seeds[0]} + TTA"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, y=y, p=p)
    return run_name, y, p


def summary(m):
    return {k: m[k] for k in KEYS} | {"per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    from src.train import load_config

    classes = load_config(args.config)["classes"]
    fixed = classes.index("Negative")
    run_name, y, p = ensemble_probs(args.config, args.seeds, args.split)
    log_p = np.log(p + EPS)
    val_json = ROOT / "results" / "metrics" / f"{run_name}_bias_val.json"

    if args.split == "val":
        before, oof = cv_gain(log_p, y, fixed)
        bias, fitted = fit_bias(log_p, y, fixed)
        print(f"val macro F1, ensemble + TTA:            {before:.4f}")
        print(f"val macro F1, offsets cross-validated:   {oof:.4f}  ({oof - before:+.4f})")
        print(f"val macro F1, offsets fitted on all val: {fitted:.4f}  (optimistic, in-sample)")
        out = {"run_name": run_name, "split": "val", "seeds": args.seeds,
               "bias": {c: float(b) for c, b in zip(classes, bias)},
               "macro_f1_before": before, "macro_f1_cv": oof, "macro_f1_in_sample": fitted,
               "before": summary(compute_metrics(y, log_p.argmax(1), classes)),
               "after_in_sample": summary(compute_metrics(y, apply_bias(log_p, bias), classes))}
    else:
        saved = json.loads(val_json.read_text())   # offsets come from val only
        bias = np.array([saved["bias"][c] for c in classes])
        before = compute_metrics(y, log_p.argmax(1), classes)
        after = compute_metrics(y, apply_bias(log_p, bias), classes)
        print(f"test macro F1, ensemble + TTA:  {before['macro_f1']:.4f}")
        print(f"test macro F1, + val offsets:   {after['macro_f1']:.4f}  "
              f"({after['macro_f1'] - before['macro_f1']:+.4f})")
        out = {"run_name": run_name, "split": "test", "seeds": args.seeds,
               "bias": saved["bias"], "before": summary(before), "after": summary(after)}
    print("offsets:", {c: round(float(b), 2) for c, b in zip(classes, bias)})
    path = ROOT / "results" / "metrics" / f"{run_name}_bias_{args.split}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
