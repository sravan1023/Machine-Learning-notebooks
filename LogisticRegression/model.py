import numpy as np


class LogisticRegression:
    """Logistic Regression classifier built from scratch using gradient descent.

    Supports binary and multi-class (one-vs-rest) classification.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6,
                 regularization=None, C=1.0, random_state=None):
        """
        Parameters
        ----------
        lr : float
            Learning rate for gradient descent.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance on the loss improvement.
        regularization : str or None
            'l2' for ridge penalty, None for no regularization.
        C : float
            Inverse of regularization strength (higher = less regularization).
        random_state : int or None
            Seed for reproducibility.
        """
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.C = C
        self.random_state = random_state

        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
        self.loss_history_ = None
        self._multi = False          # True when n_classes > 2
        self._classifiers = []       # list of (w, b) for OvR

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z):
        # numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    def _init_weights(self, n_features, rng):
        w = rng.randn(n_features) * 0.01
        b = 0.0
        return w, b

    # ------------------------------------------------------------------
    def _compute_loss(self, X, y, w, b):
        """Binary cross-entropy plus optional L2 penalty."""
        m = len(y)
        z = X @ w + b
        h = self._sigmoid(z)
        # clamp to avoid log(0)
        eps = 1e-15
        h = np.clip(h, eps, 1 - eps)
        loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

        if self.regularization == "l2":
            loss += (1.0 / (2.0 * self.C * m)) * np.dot(w, w)

        return loss

    # ------------------------------------------------------------------
    def _compute_gradients(self, X, y, w, b):
        """Gradient of the loss w.r.t. weights and bias."""
        m = len(y)
        z = X @ w + b
        h = self._sigmoid(z)
        error = h - y  # (m,)

        dw = (1.0 / m) * (X.T @ error)
        db = (1.0 / m) * np.sum(error)

        if self.regularization == "l2":
            dw += (1.0 / (self.C * m)) * w

        return dw, db

    # ------------------------------------------------------------------
    def _fit_binary(self, X, y, rng):
        """Train a single binary logistic regression model."""
        n_samples, n_features = X.shape
        w, b = self._init_weights(n_features, rng)
        loss_history = []

        for i in range(self.max_iter):
            loss = self._compute_loss(X, y, w, b)
            loss_history.append(loss)

            dw, db = self._compute_gradients(X, y, w, b)
            w -= self.lr * dw
            b -= self.lr * db

            if i > 0 and abs(loss_history[-2] - loss_history[-1]) < self.tol:
                break

        return w, b, loss_history

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit logistic regression model.

        For multi-class problems, trains one-vs-rest binary classifiers.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            self._multi = False
            # map to 0/1
            y_bin = (y == self.classes_[1]).astype(float)
            self.weights_, self.bias_, self.loss_history_ = \
                self._fit_binary(X, y_bin, rng)
        else:
            self._multi = True
            self._classifiers = []
            self.loss_history_ = []
            for cls in self.classes_:
                y_bin = (y == cls).astype(float)
                w, b, hist = self._fit_binary(X, y_bin, rng)
                self._classifiers.append((w, b))
                self.loss_history_.append(hist)

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Return probability estimates.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=float)

        if not self._multi:
            p1 = self._sigmoid(X @ self.weights_ + self.bias_)
            return np.column_stack([1 - p1, p1])
        else:
            scores = np.column_stack([
                self._sigmoid(X @ w + b) for w, b in self._classifiers
            ])
            # normalise across classes
            row_sums = scores.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return scores / row_sums

    # ------------------------------------------------------------------
    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    # ------------------------------------------------------------------
    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(np.asarray(y) == y_pred))
