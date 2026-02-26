import numpy as np


def gini(y):
    """Gini impurity for a label array.

    Parameters

    y : ndarray of shape (n,)
        Class labels.

    Returns
    -------
    float
        Gini impurity (0 = pure, approaches 1 = maximally impure).
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - float(np.sum(probs ** 2))


def entropy(y):
    """Shannon entropy for a label array.

    Parameters
    ----------
    y : ndarray of shape (n,)

    Returns
    -------
    float
        Entropy in bits (0 = pure).
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


def misclassification(y):
    """Misclassification error for a label array.

    Parameters
    ----------
    y : ndarray of shape (n,)

    Returns
    -------
    float
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return 1.0 - float(np.max(counts)) / len(y)


def information_gain(y, y_left, y_right, criterion_fn):
    """Compute the information gain of a split.

    IG = criterion(parent) - weighted average criterion(children)

    Parameters
    ----------
    y       : ndarray – parent labels.
    y_left  : ndarray – left-child labels.
    y_right : ndarray – right-child labels.
    criterion_fn : callable – gini, entropy, or misclassification.

    Returns
    -------
    float
    """
    n = len(y)
    if n == 0:
        return 0.0
    parent = criterion_fn(y)
    child = (len(y_left) / n) * criterion_fn(y_left) + \
            (len(y_right) / n) * criterion_fn(y_right)
    return parent - child


CRITERION_FUNCTIONS = {
    "gini": gini,
    "entropy": entropy,
    "misclassification": misclassification,
}
