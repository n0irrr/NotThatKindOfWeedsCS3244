"""Confusion matrix plotting and error-case extraction for error-driven iteration."""


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    raise NotImplementedError


def top_confusions(y_true, y_pred, class_names, k=5):
    """Return the k most frequent (true, predicted) misclassification pairs."""
    raise NotImplementedError
