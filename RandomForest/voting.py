import numpy as np
from collections import Counter


def majority_vote(predictions):
    """Hard majority vote across an ensemble of predictions.

    Parameters
    ----------
    predictions : list of ndarray, each of shape (n_samples,)
    Predictions from each estimator.

    Returns
    -------
    ndarray of shape (n_samples,)
    Final predicted label for each sample.
    """
    n_samples = len(predictions[0])
    final = np.empty(n_samples, dtype=predictions[0].dtype)

    for i in range(n_samples):
        votes = Counter(pred[i] for pred in predictions)
        final[i] = votes.most_common(1)[0][0]

    return final


def soft_vote(proba_list, classes):
    """Soft vote: average class probabilities, pick argmax.

    Parameters
    ----------
    proba_list : list of ndarray, each of shape (n_samples, n_classes) Probability estimates from each estimator.
    classes : ndarray of shape (n_classes,)
    Ordered class labels.

    Returns
    -------
    ndarray of shape (n_samples,) Final predicted label for each sample.
    """
    avg_proba = np.mean(proba_list, axis=0)
    return classes[np.argmax(avg_proba, axis=1)]


def weighted_vote(predictions, weights):
    """Weighted majority vote.

    Parameters
    ----------
    predictions : list of ndarray, each of shape (n_samples,)
    weights : array-like of shape (n_estimators,)
    Non-negative weight for each estimator.

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()  # normalise

    n_samples = len(predictions[0])
    final = np.empty(n_samples, dtype=predictions[0].dtype)

    for i in range(n_samples):
        vote_scores = Counter()
        for w, pred in zip(weights, predictions):
            vote_scores[pred[i]] += w
        final[i] = vote_scores.most_common(1)[0][0]

    return final
