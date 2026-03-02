import numpy as np


def bootstrap_sample(X, y, rng=None):
    """
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    rng : np.random.RandomState or None

    Returns
    -------
    X_boot : ndarray – bootstrap sample features.
    y_boot : ndarray – bootstrap sample labels.
    oob_idx : ndarray – indices of out-of-bag samples.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_samples = X.shape[0]
    indices = rng.randint(0, n_samples, size=n_samples)

    X_boot = X[indices]
    y_boot = y[indices]

    oob_mask = np.ones(n_samples, dtype=bool)
    oob_mask[np.unique(indices)] = False
    oob_idx = np.where(oob_mask)[0]

    return X_boot, y_boot, oob_idx


def bootstrap_sample_stratified(X, y, rng=None):
    """
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    rng : np.random.RandomState or None

    Returns
    -------
    X_boot, y_boot, oob_idx
    """
    if rng is None:
        rng = np.random.RandomState()

    classes = np.unique(y)
    boot_indices = []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n_cls = len(cls_idx)
        sampled = rng.choice(cls_idx, size=n_cls, replace=True)
        boot_indices.append(sampled)

    indices = np.concatenate(boot_indices)
    rng.shuffle(indices)

    X_boot = X[indices]
    y_boot = y[indices]

    oob_mask = np.ones(X.shape[0], dtype=bool)
    oob_mask[np.unique(indices)] = False
    oob_idx = np.where(oob_mask)[0]

    return X_boot, y_boot, oob_idx


def oob_score(forest, X, y):
    """Compute out-of-bag accuracy for a fitted RandomForestClassifier.

    Each sample is predicted only by trees that did NOT include it
    in their bootstrap sample.

    Parameters
    ----------
    forest : RandomForestClassifier (fitted)
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Returns
    -------
    float  –  OOB accuracy (0-1).
    """
    from collections import Counter

    n_samples = X.shape[0]
    votes = [Counter() for _ in range(n_samples)]

    for tree, oob_idx in zip(forest.estimators_, forest.oob_indices_):
        if len(oob_idx) == 0:
            continue
        preds = tree.predict(X[oob_idx])
        for idx, pred in zip(oob_idx, preds):
            votes[idx][pred] += 1

    correct = 0
    counted = 0
    for i in range(n_samples):
        if len(votes[i]) == 0:
            continue  # sample was in every bootstrap – skip
        predicted = votes[i].most_common(1)[0][0]
        if predicted == y[i]:
            correct += 1
        counted += 1

    return correct / counted if counted > 0 else 0.0
