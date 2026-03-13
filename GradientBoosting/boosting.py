import numpy as np
from tree import RegressionTreeRegressor
from loss import softmax, one_hot, multiclass_cross_entropy


class GradientBoostingClassifier:
    """Multi-class gradient boosting with regression-tree weak learners.

    Uses stage-wise additive modeling on class logits and optimizes
    multi-class cross-entropy via negative gradients.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=2, min_samples_split=2, min_samples_leaf=1,
                 max_samples=256, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.random_state = random_state

        self.classes_ = None
        self.n_classes_ = None
        self.estimators_ = []
        self.loss_history_ = []
        self._init_logits = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        self.classes_ = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y], dtype=int)
        self.n_classes_ = len(self.classes_)

        y_onehot = one_hot(y_idx, self.n_classes_)

        # Initialize logits with class priors
        priors = np.clip(y_onehot.mean(axis=0), 1e-15, 1.0)
        self._init_logits = np.log(priors)
        F = np.tile(self._init_logits, (X.shape[0], 1))

        self.estimators_ = []
        self.loss_history_ = []

        for _ in range(self.n_estimators):
            P = softmax(F)
            residuals = y_onehot - P

            stage_trees = []
            for k in range(self.n_classes_):
                if self.max_samples is None or self.max_samples >= X.shape[0]:
                    idx = np.arange(X.shape[0])
                else:
                    idx = rng.choice(X.shape[0], size=self.max_samples, replace=False)

                tree = RegressionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                )
                tree.fit(X[idx], residuals[idx, k])
                F[:, k] += self.learning_rate * tree.predict(X)
                stage_trees.append(tree)

            self.estimators_.append(stage_trees)
            self.loss_history_.append(multiclass_cross_entropy(y_onehot, softmax(F)))

        return self

    def _raw_predict_logits(self, X):
        X = np.asarray(X, dtype=float)
        if self._init_logits is None:
            raise ValueError("Model is not fitted yet. Call fit before predict.")

        logits = np.tile(self._init_logits, (X.shape[0], 1)).astype(float)

        for stage_trees in self.estimators_:
            for k, tree in enumerate(stage_trees):
                logits[:, k] += self.learning_rate * tree.predict(X)
        return logits

    def predict_proba(self, X):
        return softmax(self._raw_predict_logits(X))

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError("Model is not fitted yet. Call fit before predict.")
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
