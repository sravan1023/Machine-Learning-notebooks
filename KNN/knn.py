import numpy as np
from collections import Counter
from distances import DISTANCE_FUNCTIONS


class KNNClassifier:
    """K-Nearest Neighbors classifier built from scratch."""

    def __init__(self, k=5, metric="euclidean", weights="uniform", p=3):
        """
        Parameters
        ----------
        k : int
            Number of neighbors.
        metric : str
            Distance metric ('euclidean', 'manhattan', 'minkowski').
        weights : str
            'uniform' – equal vote, 'distance' – inverse-distance weighting.
        p : int
            Power parameter for Minkowski distance (used only when
            metric='minkowski').
        """
        self.k = k
        self.metric = metric
        self.weights = weights
        self.p = p

        self.X_train = None
        self.y_train = None
        self.classes_ = None

    # ------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    # ------------------------------------------------------------------
    def _get_distances(self, x):
        if self.metric == "minkowski":
            return DISTANCE_FUNCTIONS[self.metric](x, self.X_train, self.p)
        return DISTANCE_FUNCTIONS[self.metric](x, self.X_train)

    # ------------------------------------------------------------------
    def _predict_single(self, x):
        dists = self._get_distances(x)
        nn_idx = np.argsort(dists)[:self.k]
        nn_labels = self.y_train[nn_idx]

        if self.weights == "uniform":
            counts = Counter(nn_labels)
            return counts.most_common(1)[0][0]

        elif self.weights == "distance":
            nn_dists = dists[nn_idx]
            # if a neighbor has distance 0, assign all weight to it
            if np.any(nn_dists == 0):
                zero_mask = nn_dists == 0
                nn_labels = nn_labels[zero_mask]
                counts = Counter(nn_labels)
                return counts.most_common(1)[0][0]

            inv_dists = 1.0 / nn_dists
            class_weights = {}
            for label, w in zip(nn_labels, inv_dists):
                class_weights[label] = class_weights.get(label, 0.0) + w
            return max(class_weights, key=class_weights.get)

        else:
            raise ValueError(f"Unknown weights: {self.weights}")

    # ------------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Return class probability estimates for each sample."""
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        class_idx = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((X.shape[0], n_classes))

        for i, x in enumerate(X):
            dists = self._get_distances(x)
            nn_idx = np.argsort(dists)[:self.k]
            nn_labels = self.y_train[nn_idx]

            if self.weights == "uniform":
                for lbl in nn_labels:
                    proba[i, class_idx[lbl]] += 1.0
            else:
                nn_dists = dists[nn_idx]
                if np.any(nn_dists == 0):
                    zero_mask = nn_dists == 0
                    for lbl in nn_labels[zero_mask]:
                        proba[i, class_idx[lbl]] += 1.0
                else:
                    inv_dists = 1.0 / nn_dists
                    for lbl, w in zip(nn_labels, inv_dists):
                        proba[i, class_idx[lbl]] += w

            proba[i] /= proba[i].sum()

        return proba

    # ------------------------------------------------------------------
    def score(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))
