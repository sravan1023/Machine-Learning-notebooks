import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = float(var_smoothing)
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_prior_ = np.zeros(n_classes, dtype=float)
        self.theta_ = np.zeros((n_classes, n_features), dtype=float)
        self.var_ = np.zeros((n_classes, n_features), dtype=float)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_prior_[i] = Xc.shape[0] / X.shape[0]
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + self.var_smoothing

        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        jll = np.zeros((X.shape[0], n_classes), dtype=float)

        for i in range(n_classes):
            log_prior = np.log(self.class_prior_[i])
            var = self.var_[i]
            mean = self.theta_[i]
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var,
                axis=1,
            )
            jll[:, i] = log_prior + log_likelihood

        return jll

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        max_log = np.max(jll, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(np.sum(np.exp(jll - max_log), axis=1, keepdims=True))
        return jll - log_sum_exp

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        idx = np.argmax(jll, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))
