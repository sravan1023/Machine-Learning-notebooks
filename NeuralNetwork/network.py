import numpy as np
from layers import DenseLayer
from loss import get_loss
from optimizer import BatchGradientDescent


class NeuralNetwork:
    """Multi-layer perceptron built from scratch.

    Supports arbitrary depth, configurable activations per layer,
    and pluggable loss / optimizer.
    """

    def __init__(self, layer_sizes, activations, loss="categorical_cross_entropy",
                 optimizer=None, random_state=None):
        """
        Parameters

        layer_sizes : list of int
            Number of neurons in each layer INCLUDING the input layer.
            Example: [784, 64, 10]: 784 inputs, 64 hidden, 10 outputs.
        activations : list of str
            Activation for each hidden/output layer (len = len(layer_sizes)-1).
            Example: ['relu', 'softmax'].
        loss : str
            Loss function name ('binary_cross_entropy' or
            'categorical_cross_entropy').
        optimizer : optimizer instance or None
            Defaults to BatchGradientDescent(lr=0.01).
        random_state : int or None
            Seed for reproducibility.
        """
        assert len(activations) == len(layer_sizes) - 1, \
            "Need one activation per layer transition."

        self.loss_name = loss
        self.loss_fn, self.loss_grad = get_loss(loss)
        self.optimizer = optimizer or BatchGradientDescent(lr=0.01)
        self.random_state = random_state

        rng = np.random.RandomState(random_state)
        self.layers = []
        for i in range(len(activations)):
            seed = rng.randint(0, 2**31)
            self.layers.append(
                DenseLayer(layer_sizes[i], layer_sizes[i + 1],
                           activation=activations[i],
                           random_state=seed)
            )

        self.loss_history_ = []

  
    def _forward(self, X):
        """Full forward pass through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

  
    def _backward(self, y_true, y_pred):
        """Full backward pass (back-propagation)."""
        da = self.loss_grad(y_true, y_pred)
        for layer in reversed(self.layers):
            da = layer.backward(da)

   
    def fit(self, X, y, epochs=1000, verbose=True, print_every=100):
        """Train the network using batch gradient descent.

        Parameters

        X : ndarray of shape (m, n_features)
            Training inputs.
        y : ndarray of shape (m, n_outputs)
            Training targets (one-hot for multi-class, column vector
            for binary).
        epochs : int
            Number of training iterations.
        verbose : bool
            Print loss every `print_every` epochs.
        print_every : int
            Reporting interval.

        Returns
        -------
        self
        """
        self.loss_history_ = []

        for epoch in range(1, epochs + 1):
            # forward
            y_pred = self._forward(X)

            # compute loss
            loss = self.loss_fn(y, y_pred)
            self.loss_history_.append(loss)

            # backward
            self._backward(y, y_pred)

            # update weights
            self.optimizer.step(self.layers)

            if verbose and epoch % print_every == 0:
                print(f"  Epoch {epoch:>5d}/{epochs}  loss = {loss:.6f}")

        return self


    def predict_proba(self, X):
        """Return raw output-layer activations (probabilities).

        Parameters

        X : ndarray of shape (m, n_features)

        Returns

        ndarray of shape (m, n_outputs)
        """
        return self._forward(X)

    
    def predict(self, X):
        """Return class predictions.

        For binary (sigmoid output with 1 neuron): threshold at 0.5.
        For multi-class (softmax output): argmax.

        Parameters

        X : ndarray of shape (m, n_features)

        Returns

        ndarray of shape (m,)
        """
        proba = self.predict_proba(X)
        if proba.shape[1] == 1:
            return (proba.ravel() >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

   
    def score(self, X, y_true):
        """Classification accuracy.

        Parameters

        X : ndarray of shape (m, n_features)
        y_true : ndarray
            Integer labels (1-D) or one-hot (2-D).

        Returns

        float
        """
        y_pred = self.predict(X)
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        return float(np.mean(y_pred == y_true))


    def __repr__(self):
        arch = " -> ".join(str(l) for l in self.layers)
        return f"NeuralNetwork(\n  {arch}\n  loss='{self.loss_name}')"
