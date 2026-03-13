import numpy as np


class _RegNode:
    """Regression tree node (split or leaf)."""

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class RegressionTreeRegressor:
    """Simple CART-style regression tree with squared-error splits."""

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    @staticmethod
    def _mse(y):
        if len(y) == 0:
            return 0.0
        m = np.mean(y)
        return float(np.mean((y - m) ** 2))

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_mse = self._mse(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        for j in range(n_features):
            vals = np.unique(X[:, j])
            if len(vals) < 2:
                continue

            # Limit candidate thresholds for speed on medium-sized datasets.
            if len(vals) > 16:
                q = np.linspace(0.1, 0.9, 8)
                thresholds = np.unique(np.quantile(vals, q))
            else:
                thresholds = (vals[:-1] + vals[1:]) / 2.0

            for thr in thresholds:
                left_idx = np.where(X[:, j] <= thr)[0]
                right_idx = np.where(X[:, j] > thr)[0]

                if (len(left_idx) < self.min_samples_leaf
                        or len(right_idx) < self.min_samples_leaf):
                    continue

                left_mse = self._mse(y[left_idx])
                right_mse = self._mse(y[right_idx])
                child_mse = ((len(left_idx) * left_mse + len(right_idx) * right_mse)
                             / n_samples)
                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = thr
                    best_left = left_idx
                    best_right = right_idx

        if best_feature is None:
            return None

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left_idx": best_left,
            "right_idx": best_right,
        }

    def _build(self, X, y, depth):
        if (len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or np.allclose(y, y[0])):
            return _RegNode(value=float(np.mean(y)))

        split = self._best_split(X, y)
        if split is None:
            return _RegNode(value=float(np.mean(y)))

        left = self._build(X[split["left_idx"]], y[split["left_idx"]], depth + 1)
        right = self._build(X[split["right_idx"]], y[split["right_idx"]], depth + 1)

        return _RegNode(feature=split["feature"], threshold=split["threshold"],
                        left=left, right=right)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.tree_) for x in X], dtype=float)
