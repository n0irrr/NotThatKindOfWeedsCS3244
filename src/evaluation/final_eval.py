"""Final evaluation: pick ONE candidate by mean validation macro F1, then report its test score.

    python -m src.evaluation.final_eval [candidate ...]

Candidates default to every run in results/metrics/multiseed_summary.json. Selection looks at
validation only; the test mean ± std printed afterwards is for the selected candidate alone
and is compared with the published DeepWeeds ResNet50 result (95.7% accuracy, Olsen et al. 2019).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "metrics" / "multiseed_summary.json"
PUBLISHED_ACC = 0.957


def main(candidates):
    summary = json.loads(SUMMARY.read_text())
    cands = candidates or list(summary)
    print("candidates ranked by mean val macro F1 (test deliberately not shown):")
    for b in sorted(cands, key=lambda b: -summary[b]["val_mean"]):
        print(f"  {b:<34}{summary[b]['val_mean']:.3f} ± {summary[b]['val_std']:.3f}")
    best = max(cands, key=lambda b: summary[b]["val_mean"])
    s = summary[best]
    out = {"selected": best, "selected_by": "mean validation macro F1", "published_accuracy": PUBLISHED_ACC,
           "test_macro_f1_mean": s["test_mean"], "test_macro_f1_std": s["test_std"],
           "test_accuracy_mean": s["test_accuracy_mean"], "seeds": s["seeds"],
           "accuracy_gap_to_published": PUBLISHED_ACC - s["test_accuracy_mean"]}
    print(f"\nselected: {best}")
    print(f"test macro F1 {s['test_mean']:.3f} ± {s['test_std']:.3f}, test accuracy {s['test_accuracy_mean']:.3f} "
          f"(n={len(s['seeds'])} seeds)")
    print(f"published ResNet50 accuracy {PUBLISHED_ACC:.3f}; gap {out['accuracy_gap_to_published']:+.3f}")
    (ROOT / "results" / "metrics" / "final_evaluation.json").write_text(json.dumps(out, indent=2))
    print("saved results/metrics/final_evaluation.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="*")
    main(ap.parse_args().candidates)
