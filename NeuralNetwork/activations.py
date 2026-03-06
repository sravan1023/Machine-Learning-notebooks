import numpy as np


def relu(z):
    """Rectified Linear Unit activation.

    Parameters
    ----------
    z : ndarray
        Pre-activation values.

    Returns
    -------
    ndarray
        max(0, z) element-wise.
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU.

    Parameters
    ----------
    z : ndarray
        Pre-activation values (same z passed to relu).

    Returns
    -------
    ndarray
        1 where z > 0, else 0.
    """
    return (z > 0).astype(float)


def sigmoid(z):
    """Numerically stable sigmoid activation.

    Parameters
    ----------
    z : ndarray
        Pre-activation values.

    Returns
    -------
    ndarray
        σ(z) = 1 / (1 + exp(-z)).
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid, expressed in terms of pre-activation z.

    Parameters
    ----------
    z : ndarray
        Pre-activation values (same z passed to sigmoid).

    Returns
    -------
    ndarray
        σ(z) * (1 - σ(z)).
    """
    s = sigmoid(z)
    return s * (1.0 - s)


def softmax(z):
    """Numerically stable softmax activation (row-wise).

    Parameters
    ----------
    z : ndarray of shape (m, n_classes)
        Pre-activation logits.

    Returns
    -------
    ndarray of shape (m, n_classes)
        Probability distribution over classes for each sample.
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ── Registry for convenient lookup ───────────────────────────────────
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "softmax": (softmax, None),          # softmax gradient handled in loss
}


def get_activation(name):
    """Return (activation_fn, derivative_fn) by name.

    Parameters
    ----------
    name : str
        One of 'relu', 'sigmoid', 'softmax'.

    Returns
    -------
    tuple of (callable, callable or None)
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]
