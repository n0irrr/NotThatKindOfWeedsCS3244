"""Combine repeated runs of one config into mean +/- std, so a difference
between two models can be judged against run-to-run noise.

Runs come from `python -m src.train --config <cfg> --seed N`, saved as
results/metrics/<run_name>_s<N>.json. Then:

    python -m src.evaluation.aggregate simple_cnn_aug_geo_60ep_bf16

reads every <prefix>_s*.json and writes
    results/metrics/<prefix>_seeds.json               mean / std / min / max per metric,
                                                      per-class test F1, pooled confusion matrix
    results/confusion_matrices/<prefix>_seeds_test.png

std is the sample std (ddof=1) over runs. All runs use the same split, so it
measures training randomness (init, shuffling, augmentation), not split
variance. Every run is scored on the same test set, so the pooled matrix
(counts summed over runs, then row-normalised) equals the average of the
runs' own row-normalised matrices.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.evaluation.confusion import plot_confusion_counts

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("macro_f1", "macro_precision", "macro_recall", "accuracy")


def summarise(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "runs": [round(float(x), 4) for x in v]}


def load_runs(prefix: str, metrics_dir: Path) -> list[dict]:
    """Metrics of every <prefix>_s<N>.json, ordered by seed."""
    pattern = re.compile(rf"{re.escape(prefix)}_s(\d+)\.json")
    found = sorted((int(m.group(1)), p) for p in metrics_dir.glob(f"{prefix}_s*.json")
                   if (m := pattern.fullmatch(p.name)))
    if not found:
        raise FileNotFoundError(f"no {prefix}_s<N>.json in {metrics_dir}")
    return [json.loads(p.read_text()) for _, p in found]


def aggregate(runs: list[dict]) -> dict:
    classes = runs[0]["config"]["classes"]
    test_per_class = [r["splits"]["test"]["model"]["per_class"] for r in runs]
    if len({tuple(pc[c]["support"] for c in classes) for pc in test_per_class}) != 1:
        raise ValueError("runs were scored on different test sets, so they can't be pooled")
    missing = [r["run_name"] for r in runs if "confusion_matrix_test" not in r]
    if missing:
        raise KeyError(f"no confusion_matrix_test in {missing}: re-run them with the current src.train")

    pooled = np.sum([r["confusion_matrix_test"] for r in runs], axis=0)
    return {
        "classes": classes,
        "runs": [r["run_name"] for r in runs],
        "seeds": [r["config"]["seed"] for r in runs],
        "best_epochs": [r.get("best_epoch") for r in runs],
        "splits": {split: {k: summarise([r["splits"][split]["model"][k] for r in runs]) for k in KEYS}
                   for split in ("val", "test")},
        "per_class_test_f1": {c: summarise([pc[c]["f1"] for pc in test_per_class]) for c in classes},
        "confusion_matrix_test_pooled": pooled.tolist(),
    }


def top_pooled_confusions(summary: dict, k: int = 5) -> list[tuple[str, str, float]]:
    """k most frequent (true, predicted) mistakes, as average count per run."""
    cm = np.array(summary["confusion_matrix_test_pooled"], dtype=float) / len(summary["runs"])
    np.fill_diagonal(cm, 0)
    classes = summary["classes"]
    flat = np.argsort(cm, axis=None)[::-1][:k]
    return [(classes[t], classes[p], float(cm[t, p])) for t, p in (np.unravel_index(i, cm.shape) for i in flat)]


def main(prefix: str, root: Path = ROOT) -> dict:
    root = Path(root)
    runs = load_runs(prefix, root / "results" / "metrics")
    summary = aggregate(runs)
    name = f"{prefix}_seeds"
    out_json = root / "results" / "metrics" / f"{name}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    out_png = root / "results" / "confusion_matrices" / f"{name}_test.png"
    plot_confusion_counts(summary["confusion_matrix_test_pooled"], summary["classes"], out_png,
                          title=f"{prefix} (test, {len(runs)} seeds pooled)")

    print(f"{prefix}: {len(runs)} runs, seeds {summary['seeds']}, best epochs {summary['best_epochs']}")
    for split in ("val", "test"):
        for k in KEYS:
            s = summary["splits"][split][k]
            print(f"  {split:<4} {k:<16} {s['mean']:.4f} +/- {s['std']:.4f}   "
                  f"(min {s['min']:.4f}, max {s['max']:.4f})   runs {s['runs']}")
    print("  test per-class F1 (mean +/- std):")
    for c, s in summary["per_class_test_f1"].items():
        print(f"    {c:<16} {s['mean']:.3f} +/- {s['std']:.3f}")
    print("  top test confusions (true -> pred, mean count per run):")
    for t, p, n in top_pooled_confusions(summary):
        print(f"    {t} -> {p}: {n:.1f}")
    print(f"saved {out_json.relative_to(root)} and {out_png.relative_to(root)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="run name without the _s<N> suffix")
    main(parser.parse_args().prefix)
