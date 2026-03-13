import numpy as np


def softmax(logits):
    """Numerically stable softmax (row-wise)."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, n_classes):
    """Convert integer labels to one-hot encoded rows."""
    y = np.asarray(y).astype(int)
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def multiclass_cross_entropy(y_true, y_prob):
    """Mean multi-class cross-entropy."""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return -float(np.mean(np.sum(y_true * np.log(y_prob), axis=1)))
