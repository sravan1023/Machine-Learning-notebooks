import numpy as np
from kernel import get_kernel, kernel_matrix, kernel_vector
from optimizer import SMO


class SVC:
    """Support Vector Classifier with kernel support.

    Uses the SMO algorithm to solve the dual quadratic programme and
    supports linear, polynomial, RBF, and sigmoid kernels.
    Multi-class classification is handled via one-vs-one (OvO).
    """

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="scale",
                 coef0=0.0, tol=1e-3, max_passes=100, random_state=None):
        """
        Parameters
        ----------
        C : float
            Regularization (box constraint).
        kernel : str
            'linear', 'poly', 'rbf', or 'sigmoid'.
        degree : int
            Degree of the polynomial kernel (ignored by others).
        gamma : float or 'scale' or 'auto'
            Kernel coefficient for rbf / poly / sigmoid.
            'scale' -> 1 / (n_features * X.var()),
            'auto'  -> 1 / n_features.
        coef0 : float
            Independent term in poly / sigmoid kernels.
        tol : float
            KKT tolerance for SMO.
        max_passes : int
            SMO convergence parameter.
        random_state : int or None
            Seed for reproducibility.
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state

        # set after fit
        self.classes_ = None
        self.n_features_ = None
        self._classifiers = []  # list of binary SVMs for OvO

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _make_kernel_fn(self, X):
        gamma = self._resolve_gamma(X)
        return get_kernel(self.kernel, degree=self.degree,
                          gamma=gamma, coef0=self.coef0)


    def _fit_binary(self, X, y, kernel_fn):
        """Fit a single binary SVM (labels must be -1 / +1)."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        K = kernel_matrix(X, kernel_fn)
        smo = SMO(C=self.C, tol=self.tol, max_passes=self.max_passes)
        alphas, b = smo.solve(K, y)

        # keep only support vectors
        sv_mask = alphas > 1e-7
        return {
            "alphas": alphas[sv_mask],
            "sv_X": X[sv_mask],
            "sv_y": y[sv_mask],
            "b": b,
            "kernel_fn": kernel_fn,
            "n_sv": int(sv_mask.sum()),
        }

    def _predict_binary(self, clf, X):
        """Decision values for a fitted binary SVM."""
        alphas = clf["alphas"]
        sv_X = clf["sv_X"]
        sv_y = clf["sv_y"]
        b = clf["b"]
        kfn = clf["kernel_fn"]

        n = X.shape[0]
        dec = np.zeros(n)
        for i in range(n):
            k = kernel_vector(sv_X, X[i], kfn)
            dec[i] = np.sum(alphas * sv_y * k) + b
        return dec

    def fit(self, X, y):
        """Fit the SVM model.

        For binary problems a single SVM is trained.
        For multi-class, one-vs-one (OvO) classifiers are trained.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ndarray (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        kernel_fn = self._make_kernel_fn(X)

        self._classifiers = []

        if len(self.classes_) == 2:
            # map to -1 / +1
            pos, neg = self.classes_[1], self.classes_[0]
            y_bin = np.where(y == pos, 1.0, -1.0)
            clf = self._fit_binary(X, y_bin, kernel_fn)
            clf["pos"] = pos
            clf["neg"] = neg
            self._classifiers.append(clf)
        else:
            # one-vs-one
            for i in range(len(self.classes_)):
                for j in range(i + 1, len(self.classes_)):
                    ci, cj = self.classes_[i], self.classes_[j]
                    mask = (y == ci) | (y == cj)
                    X_ij = X[mask]
                    y_ij = np.where(y[mask] == ci, 1.0, -1.0)
                    clf = self._fit_binary(X_ij, y_ij, kernel_fn)
                    clf["pos"] = ci
                    clf["neg"] = cj
                    self._classifiers.append(clf)

        return self

    def decision_function(self, X):
        """Raw decision values."""
        X = np.asarray(X, dtype=float)
        if len(self._classifiers) == 1:
            return self._predict_binary(self._classifiers[0], X)
        raise ValueError("decision_function is only available for binary classification")


    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)

        Returns
        -------
        ndarray (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if len(self._classifiers) == 1:
            clf = self._classifiers[0]
            dec = self._predict_binary(clf, X)
            return np.where(dec >= 0, clf["pos"], clf["neg"])

        # OvO voting
        from collections import Counter
        votes = [Counter() for _ in range(n)]
        for clf in self._classifiers:
            dec = self._predict_binary(clf, X)
            for i in range(n):
                winner = clf["pos"] if dec[i] >= 0 else clf["neg"]
                votes[i][winner] += 1

        return np.array([v.most_common(1)[0][0] for v in votes])


    def score(self, X, y):
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_n_support(self):
        """Total number of support vectors across all sub-classifiers."""
        return sum(clf["n_sv"] for clf in self._classifiers)

    def get_params(self):
        return {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
        }
