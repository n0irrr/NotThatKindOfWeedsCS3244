"""Confusion matrix plotting and error-case extraction for error-driven iteration."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, title=None):
    """Row-normalised confusion matrix (rows = true class, i.e. recall on diagonal)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def top_confusions(y_true, y_pred, class_names, k=5):
    """Return the k most frequent (true, predicted) misclassification pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    pairs = [np.unravel_index(i, cm.shape) for i in flat]
    return [(class_names[t], class_names[p], int(cm[t, p])) for t, p in pairs if cm[t, p] > 0]
