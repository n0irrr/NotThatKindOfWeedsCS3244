"""Mean ± std over seeds.

    python -m src.evaluation.aggregate simple_cnn_aug_geo_60ep resnet50_aug_geo_30ep ...

For each base run name, collects results/metrics/{base}_s{seed}.json (written by
`python -m src.train --config ... --seed N`) and reports val / test macro F1 and
per-class test F1 as mean ± sample std (ddof=1). With exactly two candidates it
also runs a Welch t-test on test macro F1. Writes results/metrics/multiseed_summary.json.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "metrics"
SUMMARY = METRICS / "multiseed_summary.json"


def collect(base: str) -> dict:
    files = sorted(p for p in METRICS.glob(f"{base}_s*.json") if re.fullmatch(rf"{re.escape(base)}_s\d+", p.stem))
    if not files:
        raise SystemExit(f"no seeded results for {base!r} in {METRICS} (run src.train with --seed)")
    runs = [json.loads(p.read_text()) for p in files]
    val = [r["splits"]["val"]["model"]["macro_f1"] for r in runs]
    test = [r["splits"]["test"]["model"]["macro_f1"] for r in runs]
    classes = list(runs[0]["splits"]["test"]["model"]["per_class"])
    per_class = {c: [r["splits"]["test"]["model"]["per_class"][c]["f1"] for r in runs] for c in classes}
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    return {
        "seeds": [r["config"]["seed"] for r in runs],
        "val_macro_f1": val, "test_macro_f1": test,
        "val_mean": float(np.mean(val)), "val_std": sd(val),
        "test_mean": float(np.mean(test)), "test_std": sd(test),
        "test_accuracy_mean": float(np.mean([r["splits"]["test"]["model"]["accuracy"] for r in runs])),
        "per_class_test_f1": {c: {"mean": float(np.mean(v)), "std": sd(v)} for c, v in per_class.items()},
    }


def main(bases):
    summary = {b: collect(b) for b in bases}
    print(f"{'run':<34}{'n':>3}{'val macro F1':>20}{'test macro F1':>20}")
    for b, s in summary.items():
        print(f"{b:<34}{len(s['seeds']):>3}{s['val_mean']:>12.3f} ± {s['val_std']:.3f}"
              f"{s['test_mean']:>12.3f} ± {s['test_std']:.3f}")
    classes = list(next(iter(summary.values()))["per_class_test_f1"])
    print("\ntest F1 per class (mean ± std)")
    print(f"{'':<16}" + "".join(f"{b[:26]:>28}" for b in summary))
    for c in classes:
        print(f"{c:<16}" + "".join(
            f"{s['per_class_test_f1'][c]['mean']:>18.3f} ± {s['per_class_test_f1'][c]['std']:.3f}" for s in summary.values()))
    if len(summary) == 2:
        from scipy.stats import ttest_ind
        a, b = summary.values()
        for split in ("val", "test"):
            t, p = ttest_ind(a[f"{split}_macro_f1"], b[f"{split}_macro_f1"], equal_var=False)
            print(f"Welch t-test on {split} macro F1: t={t:.2f}, p={p:.3f} (n={len(a['seeds'])} vs {len(b['seeds'])})")
    old = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    SUMMARY.write_text(json.dumps({**old, **summary}, indent=2))
    print(f"\nsaved {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="base run names (config run_name without the _s{seed} suffix)")
    main(ap.parse_args().runs)
