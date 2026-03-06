import numpy as np


class BatchGradientDescent:
    """Vanilla (batch) gradient descent optimizer for neural networks.

    Updates every layer's weights and biases using the full-batch
    gradients computed during back-propagation.
    """

    def __init__(self, lr=0.01):
        """
        Parameters
        ----------
        lr : float
            Learning rate (step size).
        """
        self.lr = lr

    def step(self, layers):
        """Perform a single parameter update across all layers.

        Parameters
        ----------
        layers : list of DenseLayer
            Network layers whose .dW and .db have been computed.
        """
        for layer in layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


class SGD:
    """Stochastic / mini-batch gradient descent optimizer.

    Shuffles data each epoch and updates parameters after each
    mini-batch forward-backward pass.
    """

    def __init__(self, lr=0.01, batch_size=32, random_state=None):
        """
        Parameters
        ----------
        lr : float
            Learning rate.
        batch_size : int
            Mini-batch size.  Use 1 for pure SGD.
        random_state : int or None
            Seed for reproducible shuffling.
        """
        self.lr = lr
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)

    def step(self, layers):
        """Single parameter update (same interface as BatchGradientDescent)."""
        for layer in layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

    def get_batches(self, X, y):
        """Yield (X_batch, y_batch) mini-batches for one epoch.

        Parameters
        ----------
        X : ndarray of shape (m, ...)
        y : ndarray of shape (m, ...)

        Yields
        ------
        (X_batch, y_batch)
        """
        m = len(y)
        indices = np.arange(m)
        self.rng.shuffle(indices)
        for start in range(0, m, self.batch_size):
            idx = indices[start:start + self.batch_size]
            yield X[idx], y[idx]


class MomentumSGD:
    """SGD with momentum for faster convergence."""

    def __init__(self, lr=0.01, momentum=0.9):
        """
        Parameters
        ----------
        lr : float
            Learning rate.
        momentum : float
            Momentum coefficient (typically 0.9).
        """
        self.lr = lr
        self.momentum = momentum
        self._velocities = {}

    def step(self, layers):
        """Update parameters with momentum."""
        for i, layer in enumerate(layers):
            if i not in self._velocities:
                self._velocities[i] = {
                    "vW": np.zeros_like(layer.W),
                    "vb": np.zeros_like(layer.b),
                }
            v = self._velocities[i]
            v["vW"] = self.momentum * v["vW"] - self.lr * layer.dW
            v["vb"] = self.momentum * v["vb"] - self.lr * layer.db
            layer.W += v["vW"]
            layer.b += v["vb"]
