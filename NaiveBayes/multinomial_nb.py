import numpy as np


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.class_count_ = None
        self.class_log_prior_ = None
        self.feature_count_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        X = np.maximum(X, 0.0)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_count_ = np.zeros(n_classes, dtype=float)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=float)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_count_[i] = Xc.shape[0]
            self.feature_count_[i] = Xc.sum(axis=0)

        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        X = np.maximum(X, 0.0)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

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
