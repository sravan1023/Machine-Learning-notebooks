import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob):
    """Compute binary cross-entropy loss.

    Parameters
    ----------
    y_true : ndarray of shape (m,)
        Ground-truth labels (0 or 1).
    y_prob : ndarray of shape (m,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Mean binary cross-entropy.
    """
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_prob)
                          + (1 - y_true) * np.log(1 - y_prob)))


def cross_entropy(y_true, y_prob):
    """Multi-class cross-entropy loss (one-hot y_true or integer labels).

    Parameters
    ----------
    y_true : ndarray of shape (m,) or (m, n_classes)
        If 1-D, contains integer class labels.
    y_prob : ndarray of shape (m, n_classes)
        Predicted class probabilities.

    Returns
    -------
    float
        Mean cross-entropy.
    """
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    if y_true.ndim == 1:
        # integer labels → one-hot
        n_classes = y_prob.shape[1]
        one_hot = np.zeros_like(y_prob)
        one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
        y_true = one_hot

    return -float(np.mean(np.sum(y_true * np.log(y_prob), axis=1)))


def l2_penalty(weights, C):
    """L2 regularization term (ridge).

    Parameters
    ----------
    weights : ndarray
        Weight vector or list of weight vectors.
    C : float
        Inverse regularization strength.

    Returns
    -------
    float
        Penalty value  (1 / 2C) * ||w||^2.
    """
    if isinstance(weights, list):
        return sum((1.0 / (2.0 * C)) * np.dot(w, w) for w in weights)
    return (1.0 / (2.0 * C)) * float(np.dot(weights, weights))
