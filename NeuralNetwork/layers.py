import numpy as np
from activations import get_activation


class DenseLayer:
    """Fully-connected (dense) layer with configurable activation.

    Stores pre-activation z, post-activation a, and gradients
    for use during back-propagation.
    """

    def __init__(self, n_in, n_out, activation="relu", random_state=None):
        """
        Parameters
        ----------
        n_in : int
            Number of input features.
        n_out : int
            Number of neurons in the layer.
        activation : str
            Activation function name ('relu', 'sigmoid', 'softmax').
        random_state : int or None
            Seed for weight initialisation.
        """
        rng = np.random.RandomState(random_state)

        # He initialisation for ReLU, Xavier for others
        if activation == "relu":
            scale = np.sqrt(2.0 / n_in)
        else:
            scale = np.sqrt(1.0 / n_in)

        self.W = rng.randn(n_in, n_out) * scale
        self.b = np.zeros((1, n_out))

        self.activation_name = activation
        self.act_fn, self.act_deriv = get_activation(activation)

        # cache (populated during forward pass)
        self.z = None      # pre-activation
        self.a = None      # post-activation (output)
        self.input = None  # input to this layer

        # gradients (populated during backward pass)
        self.dW = None
        self.db = None

    def forward(self, X):
        """Compute the layer output.

        Parameters
        ----------
        X : ndarray of shape (m, n_in)
            Input matrix.

        Returns
        -------
        ndarray of shape (m, n_out)
            Activated output.
        """
        self.input = X
        self.z = X @ self.W + self.b        # (m, n_out)
        self.a = self.act_fn(self.z)         # (m, n_out)
        return self.a


    def backward(self, da):
        """Compute gradients and propagate error to previous layer.

        Parameters
        ----------
        da : ndarray of shape (m, n_out)
            Gradient of the loss w.r.t. this layer's activation (or
            combined loss-softmax gradient δ for the output layer).

        Returns
        -------
        ndarray of shape (m, n_in)
            Gradient of the loss w.r.t. the layer's input.
        """
        m = self.input.shape[0]

        if self.activation_name == "softmax":
            # softmax + cross-entropy gradient passed directly as da = (a - y)
            dz = da
        else:
            dz = da * self.act_deriv(self.z)   # element-wise

        self.dW = (1.0 / m) * (self.input.T @ dz)
        self.db = (1.0 / m) * np.sum(dz, axis=0, keepdims=True)

        da_prev = dz @ self.W.T
        return da_prev

    def __repr__(self):
        n_in, n_out = self.W.shape
        return (f"DenseLayer(n_in={n_in}, n_out={n_out}, "
                f"activation='{self.activation_name}')")
