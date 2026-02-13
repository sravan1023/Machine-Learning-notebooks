import numpy as np


def inertia(X, labels, centroids):
    return sum(np.sum((X[labels == k] - centroids[k]) ** 2)
               for k in range(len(centroids)))


def silhouette_score(X, labels):
    n = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    scores = np.zeros(n)

    for i in range(n):
        own_cluster = labels[i]
        own_mask = labels == own_cluster

        if own_mask.sum() > 1:
            a_i = np.mean(np.sqrt(np.sum((X[own_mask] - X[i]) ** 2, axis=1)))
        else:
            a_i = 0.0

        b_i = np.inf
        for k in unique_labels:
            if k == own_cluster:
                continue
            other_mask = labels == k
            mean_dist = np.mean(np.sqrt(np.sum((X[other_mask] - X[i]) ** 2, axis=1)))
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(scores))
