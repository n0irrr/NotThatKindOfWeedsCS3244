"""Macro precision / recall / F1, per-class breakdown, majority-class baseline."""


def compute_metrics(y_true, y_pred, class_names) -> dict:
    raise NotImplementedError


def majority_class_baseline(y_true) -> dict:
    raise NotImplementedError
