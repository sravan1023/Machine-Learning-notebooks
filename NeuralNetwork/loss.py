import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss.

    Parameters
    ----------
    y_true : ndarray of shape (m, 1)
        Ground-truth labels (0 or 1).
    y_pred : ndarray of shape (m, 1)
        Predicted probabilities (sigmoid output).

    Returns
    -------
    float
        Mean binary cross-entropy.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_pred)
                          + (1 - y_true) * np.log(1 - y_pred)))


def binary_cross_entropy_gradient(y_true, y_pred):
    """Gradient of binary cross-entropy w.r.t. sigmoid activations.

    Parameters
    ----------
    y_true : ndarray of shape (m, 1)
    y_pred : ndarray of shape (m, 1)

    Returns
    -------
    ndarray of shape (m, 1)
        ∂L/∂a = -(y/a) + (1-y)/(1-a).
        The layer backward pass multiplies by sigmoid'(z) = a(1-a),
        which simplifies the full chain to (a - y).
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)


def categorical_cross_entropy(y_true, y_pred):
    """Multi-class cross-entropy loss.

    Parameters
    ----------
    y_true : ndarray of shape (m, n_classes)
        One-hot encoded ground-truth labels.
    y_pred : ndarray of shape (m, n_classes)
        Predicted class probabilities (softmax output).

    Returns
    -------
    float
        Mean cross-entropy.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -float(np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


def categorical_cross_entropy_gradient(y_true, y_pred):
    """Combined softmax + cross-entropy gradient.

    When using softmax output with cross-entropy, the gradient
    simplifies to (ŷ - y).

    Parameters
    ----------
    y_true : ndarray of shape (m, n_classes)
        One-hot encoded labels.
    y_pred : ndarray of shape (m, n_classes)
        Softmax output.

    Returns
    -------
    ndarray of shape (m, n_classes)
    """
    return y_pred - y_true


# ── Registry for convenient lookup ───────────────────────────────────
LOSSES = {
    "binary_cross_entropy": (binary_cross_entropy,
                             binary_cross_entropy_gradient),
    "categorical_cross_entropy": (categorical_cross_entropy,
                                  categorical_cross_entropy_gradient),
}


def get_loss(name):
    """Return (loss_fn, gradient_fn) by name.

    Parameters
    ----------
    name : str
        One of 'binary_cross_entropy', 'categorical_cross_entropy'.

    Returns
    -------
    tuple of (callable, callable)
    """
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. "
                         f"Choose from {list(LOSSES.keys())}")
    return LOSSES[name]
