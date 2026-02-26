import numpy as np
from criteria import CRITERION_FUNCTIONS, information_gain


def best_split(X, y, criterion="gini", min_samples_leaf=1):
    """
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    criterion : str
        Impurity criterion ('gini', 'entropy', 'misclassification').
    min_samples_leaf : int
        Minimum number of samples required in each child node.

    Returns
    -------
    dict or None
        {'feature': int, 'threshold': float, 'gain': float,
         'left_idx': ndarray, 'right_idx': ndarray}
        None if no valid split exists.
    """
    criterion_fn = CRITERION_FUNCTIONS[criterion]
    n_samples, n_features = X.shape

    best = None
    best_gain = -np.inf

    for feat in range(n_features):
        values = X[:, feat]
        thresholds = np.unique(values)

        for thresh in thresholds:
            left_mask = values <= thresh
            right_mask = ~left_mask

            n_left = int(np.sum(left_mask))
            n_right = int(np.sum(right_mask))

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            gain = information_gain(y, y[left_mask], y[right_mask],
                                    criterion_fn)

            if gain > best_gain:
                best_gain = gain
                best = {
                    "feature": feat,
                    "threshold": thresh,
                    "gain": gain,
                    "left_idx": np.where(left_mask)[0],
                    "right_idx": np.where(right_mask)[0],
                }

    return best


def best_split_random(X, y, criterion="gini", min_samples_leaf=1,
                      max_features=None, rng=None):
    """Best split over a random subset of features (for random-forest use).

    Parameters
    ----------
    max_features : int or None
        Number of features to consider.  None = all features.
    rng : np.random.RandomState or None

    Returns
    -------
    dict or None
    """
    criterion_fn = CRITERION_FUNCTIONS[criterion]
    n_samples, n_features = X.shape

    if max_features is None or max_features >= n_features:
        feat_subset = np.arange(n_features)
    else:
        if rng is None:
            rng = np.random.RandomState()
        feat_subset = rng.choice(n_features, max_features, replace=False)

    best = None
    best_gain = -np.inf

    for feat in feat_subset:
        values = X[:, feat]
        thresholds = np.unique(values)

        for thresh in thresholds:
            left_mask = values <= thresh
            right_mask = ~left_mask

            n_left = int(np.sum(left_mask))
            n_right = int(np.sum(right_mask))

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            gain = information_gain(y, y[left_mask], y[right_mask],
                                    criterion_fn)

            if gain > best_gain:
                best_gain = gain
                best = {
                    "feature": feat,
                    "threshold": thresh,
                    "gain": gain,
                    "left_idx": np.where(left_mask)[0],
                    "right_idx": np.where(right_mask)[0],
                }

    return best
