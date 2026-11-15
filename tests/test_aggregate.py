"""src.evaluation.aggregate: mean/std over seeds and the pooled confusion matrix."""
import json

import numpy as np
import pytest

from src.evaluation import aggregate

CLASSES = ["a", "b"]


def fake_run(seed, f1, cm, prefix="model"):
    cm = np.asarray(cm)
    per_class = {c: {"precision": f1, "recall": f1, "f1": f1, "support": int(cm[i].sum())}
                 for i, c in enumerate(CLASSES)}
    scores = {"accuracy": f1, "macro_precision": f1, "macro_recall": f1, "macro_f1": f1,
              "per_class": per_class}
    return {"run_name": f"{prefix}_s{seed}", "config": {"seed": seed, "classes": CLASSES},
            "best_epoch": 3, "splits": {"val": {"model": scores}, "test": {"model": scores}},
            "confusion_matrix_test": cm.tolist()}


def write_runs(root, runs):
    metrics = root / "results" / "metrics"
    metrics.mkdir(parents=True)
    for r in runs:
        (metrics / f"{r['run_name']}.json").write_text(json.dumps(r))


def test_mean_std_and_pooled_matrix(tmp_path):
    write_runs(tmp_path, [fake_run(1, 0.6, [[3, 1], [0, 2]]),
                          fake_run(2, 0.8, [[4, 0], [1, 1]]),
                          fake_run(10, 0.7, [[2, 2], [1, 1]]),
                          fake_run(1, 0.1, [[4, 0], [0, 2]], prefix="model_other")])  # not this prefix
    summary = aggregate.main("model", root=tmp_path)

    assert summary["seeds"] == [1, 2, 10]
    f1 = summary["splits"]["test"]["macro_f1"]
    assert f1["mean"] == pytest.approx(0.7)
    assert f1["std"] == pytest.approx(0.1)   # sample std of 0.6, 0.8, 0.7
    assert summary["confusion_matrix_test_pooled"] == [[9, 3], [2, 4]]
    assert (tmp_path / "results" / "confusion_matrices" / "model_seeds_test.png").exists()


def test_refuses_different_test_sets(tmp_path):
    write_runs(tmp_path, [fake_run(1, 0.6, [[3, 1], [0, 2]]), fake_run(2, 0.8, [[5, 0], [1, 1]])])
    with pytest.raises(ValueError, match="different test sets"):
        aggregate.main("model", root=tmp_path)
