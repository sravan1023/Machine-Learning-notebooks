import numpy as np
from loss import sigmoid


class GradientDescent:
    """Vanilla (batch) gradient descent optimizer for logistic regression."""

    def __init__(self, lr=0.01):
        """
        Parameters
        ----------
        lr : float
            Learning rate (step size).
        """
        self.lr = lr

    def step(self, w, b, dw, db):
        """Perform a single parameter update.

        Parameters
        ----------
        w  : ndarray  – current weight vector.
        b  : float    – current bias.
        dw : ndarray  – gradient w.r.t. weights.
        db : float    – gradient w.r.t. bias.

        Returns
        -------
        w_new, b_new
        """
        w_new = w - self.lr * dw
        b_new = b - self.lr * db
        return w_new, b_new


class SGD:
    """Stochastic / mini-batch gradient descent optimizer."""

    def __init__(self, lr=0.01, batch_size=32, random_state=None):
        """
        Parameters
        ----------
        lr : int
            Learning rate.
        batch_size : int
            Mini-batch size.  Use 1 for pure SGD, len(X) for batch GD.
        random_state : int or None
            Seed for shuffling.
        """
        self.lr = lr
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)

    def step(self, w, b, dw, db):
        """Single parameter update (same interface as GradientDescent)."""
        w_new = w - self.lr * dw
        b_new = b - self.lr * db
        return w_new, b_new

    def get_batches(self, X, y):
        """Yield (X_batch, y_batch) mini-batches for one epoch."""
        m = len(y)
        indices = np.arange(m)
        self.rng.shuffle(indices)
        for start in range(0, m, self.batch_size):
            idx = indices[start:start + self.batch_size]
            yield X[idx], y[idx]


def compute_gradients(X, y, w, b, regularization=None, C=1.0):
    """Compute gradients for binary logistic regression.

    Parameters
    ----------
    X : ndarray (m, n)
    y : ndarray (m,)  — binary labels 0/1.
    w : ndarray (n,)
    b : float
    regularization : str or None   ('l2' or None)
    C : float   Inverse regularization strength.

    Returns
    -------
    dw, db
    """
    m = len(y)
    h = sigmoid(X @ w + b)
    error = h - y

    dw = (1.0 / m) * (X.T @ error)
    db = (1.0 / m) * np.sum(error)

    if regularization == "l2":
        dw += (1.0 / (C * m)) * w

    return dw, db
