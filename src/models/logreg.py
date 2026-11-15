"""Logistic regression baseline on classical features (second classical learner,
so a weak result can be blamed on the representation or on KNN separately)."""
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build(params: dict):
    """StandardScaler -> optional PCA -> multinomial logistic regression.
    params: C (inverse L2 strength), pca (n_components or null), max_iter, seed."""
    steps = [("scale", StandardScaler())]
    if params.get("pca"):
        steps.append(("pca", PCA(n_components=params["pca"], random_state=params.get("seed"))))
    steps.append(("logreg", LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=params.get("max_iter", 2000),
        random_state=params.get("seed"),
    )))
    return Pipeline(steps)
