"""Macro precision / recall / F1, per-class breakdown, majority-class baseline."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, class_names) -> dict:
    labels = list(range(len(class_names)))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "per_class": {
            name: {"precision": float(pc[i]), "recall": float(rc[i]),
                   "f1": float(fc[i]), "support": int(sc[i])}
            for i, name in enumerate(class_names)
        },
    }


def majority_class_baseline(y_true, y_train=None, class_names=None) -> dict:
    """Predict the most frequent class (of y_train if given, else y_true)."""
    y_true = np.asarray(y_true)
    ref = np.asarray(y_train) if y_train is not None else y_true
    majority = np.bincount(ref).argmax()
    if class_names is None:
        class_names = [str(i) for i in range(max(ref.max(), y_true.max()) + 1)]
    return compute_metrics(y_true, np.full_like(y_true, majority), class_names)
