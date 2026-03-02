import sys
import os
import numpy as np

# allow importing DecisionTree modules from the sibling folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DecisionTree"))

from tree import DecisionTreeClassifier
from bootstrap import bootstrap_sample, oob_score as _oob_score
from voting import majority_vote, soft_vote


class RandomForestClassifier:
    """Random-forest classifier built on top of DecisionTreeClassifier.

    Each tree is trained on a bootstrap sample of the data and considers
    only a random subset of features at every split (controlled by
    ``max_features``).
    """

    def __init__(self, n_estimators=100, criterion="gini",
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", bootstrap=True,
                 oob_score=False, random_state=None):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        criterion : str
            Impurity measure ('gini', 'entropy', 'misclassification').
        max_depth : int or None
            Maximum depth of each tree.
        min_samples_split : int
            Minimum samples required to split a node.
        min_samples_leaf : int
            Minimum samples required in each leaf.
        max_features : int, str or None
            Number of features to consider at each split.
            'sqrt' → sqrt(n_features), 'log2' → log2(n_features),
            int → exact number, None → all features.
        bootstrap : bool
            Whether to use bootstrap sampling.  If False, each tree
            sees the full dataset.
        oob_score : bool
            Whether to compute the out-of-bag accuracy after fitting.
        random_state : int or None
            Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        # set after fit
        self.estimators_ = []
        self.oob_indices_ = []
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.oob_score_ = None

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(f"Invalid max_features: {self.max_features}")

    def fit(self, X, y):
        """Build the forest.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        mf = self._resolve_max_features(self.n_features_)
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        self.oob_indices_ = []
        importances = np.zeros(self.n_features_)

        for _ in range(self.n_estimators):
            seed = rng.randint(0, 2**31)
            tree_rng = np.random.RandomState(seed)

            if self.bootstrap:
                X_boot, y_boot, oob_idx = bootstrap_sample(X, y, rng=tree_rng)
            else:
                X_boot, y_boot, oob_idx = X, y, np.array([], dtype=int)

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                random_state=seed,
            )
            tree.fit(X_boot, y_boot)

            self.estimators_.append(tree)
            self.oob_indices_.append(oob_idx)

            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        # average feature importances
        total = importances.sum()
        if total > 0:
            self.feature_importances_ = importances / total
        else:
            self.feature_importances_ = importances

        # out-of-bag score
        if self.oob_score and self.bootstrap:
            self.oob_score_ = _oob_score(self, X, y)

        return self

    def predict(self, X):
        """Predict class labels using hard majority vote."""
        X = np.asarray(X, dtype=float)
        predictions = [tree.predict(X) for tree in self.estimators_]
        return majority_vote(predictions)

    def predict_proba(self, X):
        """Average predicted class probabilities across all trees."""
        X = np.asarray(X, dtype=float)
        proba_list = [tree.predict_proba(X) for tree in self.estimators_]
        return np.mean(proba_list, axis=0)

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }
