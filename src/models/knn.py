"""KNN baseline on classical features."""
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build(params: dict):
    """StandardScaler -> optional PCA -> KNN. params: n_neighbors, weights,
    metric, pca (n_components or null), seed."""
    steps = [("scale", StandardScaler())]
    if params.get("pca"):
        steps.append(("pca", PCA(n_components=params["pca"], random_state=params.get("seed"))))
    steps.append(("knn", KNeighborsClassifier(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "uniform"),
        metric=params.get("metric", "euclidean"),
        n_jobs=-1,
    )))
    return Pipeline(steps)
