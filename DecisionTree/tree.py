import numpy as np
from collections import Counter
from splitter import best_split


class _Node:
    """Internal tree node (decision or leaf)."""

    __slots__ = ("feature", "threshold", "left", "right",
                 "value", "depth", "n_samples", "impurity")

    def __init__(self, *, feature=None, threshold=None,
                 left=None, right=None, value=None,
                 depth=0, n_samples=0, impurity=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # majority class (leaf only)
        self.depth = depth
        self.n_samples = n_samples
        self.impurity = impurity

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """Decision-tree classifier"""

    def __init__(self, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        """
        Parameters
        ----------
        criterion : str
            Impurity criterion ('gini', 'entropy', 'misclassification').
        max_depth : int or None
            Maximum depth of the tree.  None = grow until pure / min limits.
        min_samples_split : int
            Minimum samples required to split an internal node.
        min_samples_leaf : int
            Minimum samples required in each child after a split.
        max_features : int or None
            Number of features to consider per split (None = all).
        random_state : int or None
            Seed for reproducibility (used only when max_features is set).
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self._importance_accum = None

    @staticmethod
    def _majority_class(y):
        counts = Counter(y)
        return counts.most_common(1)[0][0]


    def _build(self, X, y, depth):
        from criteria import CRITERION_FUNCTIONS
        criterion_fn = CRITERION_FUNCTIONS[self.criterion]
        impurity = criterion_fn(y)
        n_samples = len(y)

        # stopping conditions
        if (impurity == 0.0
                or n_samples < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return _Node(value=self._majority_class(y),
                         depth=depth, n_samples=n_samples, impurity=impurity)

        split = best_split(X, y, criterion=self.criterion,
                           min_samples_leaf=self.min_samples_leaf)

        if split is None:
            return _Node(value=self._majority_class(y),
                         depth=depth, n_samples=n_samples, impurity=impurity)

        # accumulate feature importance (weighted impurity decrease)
        self._importance_accum[split["feature"]] += \
            (n_samples / self._n_total) * split["gain"]

        left_child = self._build(X[split["left_idx"]],
                                 y[split["left_idx"]], depth + 1)
        right_child = self._build(X[split["right_idx"]],
                                  y[split["right_idx"]], depth + 1)

        return _Node(feature=split["feature"],
                     threshold=split["threshold"],
                     left=left_child, right=right_child,
                     depth=depth, n_samples=n_samples, impurity=impurity)


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._n_total = len(y)
        self._importance_accum = np.zeros(self.n_features_)

        self.tree_ = self._build(X, y, depth=0)

        # normalise feature importances
        total = self._importance_accum.sum()
        if total > 0:
            self.feature_importances_ = self._importance_accum / total
        else:
            self.feature_importances_ = self._importance_accum

        return self

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x, self.tree_) for x in X])

    def predict_proba(self, X):
        """Return class probability estimates (fraction of training
        samples of each class that fall into the same leaf)."""
        X = np.asarray(X, dtype=float)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        class_idx = {c: i for i, c in enumerate(self.classes_)}

        for i, x in enumerate(X):
            node = self.tree_
            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # leaf reached — count classes that fell here during training
            # (simple: assign 1.0 to the majority class)
            proba[i, class_idx[node.value]] = 1.0

        return proba


    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(np.asarray(y) == y_pred))

    def get_depth(self):
        """Return the depth of the tree."""
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.tree_)

    def get_n_leaves(self):
        """Return the number of leaf nodes."""
        def _count(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.tree_)

    # ------------------------------------------------------------------
    def print_tree(self, feature_names=None, indent="  "):
        """Print a text representation of the tree."""
        def _show(node, prefix=""):
            if node.is_leaf:
                print(f"{prefix}class = {node.value}  "
                      f"(n={node.n_samples}, impurity={node.impurity:.4f})")
                return
            fname = (feature_names[node.feature]
                     if feature_names else f"X[{node.feature}]")
            print(f"{prefix}{fname} <= {node.threshold:.4f}  "
                  f"(n={node.n_samples}, impurity={node.impurity:.4f})")
            _show(node.left, prefix + indent + "├─ ")
            _show(node.right, prefix + indent + "└─ ")

        if self.tree_ is not None:
            _show(self.tree_)
