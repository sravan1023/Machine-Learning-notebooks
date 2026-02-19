import numpy as np


def euclidean(x, X):
    """Euclidean distance from vector x to every row in X."""
    return np.sqrt(np.sum((X - x) ** 2, axis=1))


def manhattan(x, X):
    """Manhattan (L1) distance from vector x to every row in X."""
    return np.sum(np.abs(X - x), axis=1)


def minkowski(x, X, p=3):
    """Minkowski distance of order p from vector x to every row in X."""
    return np.sum(np.abs(X - x) ** p, axis=1) ** (1.0 / p)


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "minkowski": minkowski,
}
