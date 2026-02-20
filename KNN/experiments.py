import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from knn import KNNClassifier
from metrics import accuracy, precision_recall_f1


# ── Dataset helpers ──────────────────────────────────────────────────
def load_datasets(test_size=0.3, random_state=42):
    datasets = {}
    for loader, name in [(load_iris, "Iris"),
                         (load_wine, "Wine"),
                         (load_breast_cancer, "Breast Cancer")]:
        data = loader()
        X = StandardScaler().fit_transform(data.data)
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        datasets[name] = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "target_names": list(data.target_names),
        }
    return datasets


# ── Sweep k values ───────────────────────────────────────────────────
def sweep_k(X_train, y_train, X_test, y_test,
            k_range=range(1, 26), metric="euclidean", weights="uniform"):
    train_accs, test_accs = [], []
    for k in k_range:
        knn = KNNClassifier(k=k, metric=metric, weights=weights)
        knn.fit(X_train, y_train)
        train_accs.append(knn.score(X_train, y_train))
        test_accs.append(knn.score(X_test, y_test))
    return list(k_range), train_accs, test_accs


# ── Compare distance metrics ────────────────────────────────────────
def compare_metrics(X_train, y_train, X_test, y_test,
                    k=5, metrics=("euclidean", "manhattan", "minkowski")):
    results = {}
    for m in metrics:
        knn = KNNClassifier(k=k, metric=m)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy(y_test, preds)
        prf = precision_recall_f1(y_test, preds)
        results[m] = {"accuracy": acc, "macro_f1": prf["macro"]["f1"]}
    return results


# ── Compare weighting schemes ───────────────────────────────────────
def compare_weights(X_train, y_train, X_test, y_test,
                    k_range=range(1, 26)):
    results = {}
    for w in ("uniform", "distance"):
        _, _, test_accs = sweep_k(X_train, y_train, X_test, y_test,
                                  k_range=k_range, weights=w)
        results[w] = test_accs
    return list(k_range), results


# ── Cross-validation (manual k-fold) ────────────────────────────────
def cross_validate(X, y, k=5, n_folds=5, metric="euclidean",
                   weights="uniform", random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_accs = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        knn = KNNClassifier(k=k, metric=metric, weights=weights)
        knn.fit(X[train_idx], y[train_idx])
        fold_accs.append(knn.score(X[val_idx], y[val_idx]))

    return np.array(fold_accs)


# ── Best-k via cross-validation ─────────────────────────────────────
def best_k_cv(X, y, k_range=range(1, 26), n_folds=5, metric="euclidean",
              weights="uniform", random_state=42):
    mean_accs = []
    for k in k_range:
        accs = cross_validate(X, y, k=k, n_folds=n_folds,
                              metric=metric, weights=weights,
                              random_state=random_state)
        mean_accs.append(accs.mean())
    best_idx = int(np.argmax(mean_accs))
    return list(k_range), mean_accs, list(k_range)[best_idx]


# ── Run everything ───────────────────────────────────────────────────
def run_all():
    datasets = load_datasets()
    all_results = {}

    for name, d in datasets.items():
        X_tr, X_te = d["X_train"], d["X_test"]
        y_tr, y_te = d["y_train"], d["y_test"]
        target_names = d["target_names"]

        # k sweep
        k_vals, train_accs, test_accs = sweep_k(X_tr, y_tr, X_te, y_te)
        best_k_idx = int(np.argmax(test_accs))
        best_k = k_vals[best_k_idx]

        # cross-validation
        cv_k_vals, cv_mean_accs, cv_best_k = best_k_cv(
            np.vstack([X_tr, X_te]),
            np.concatenate([y_tr, y_te]))

        # best model
        knn_best = KNNClassifier(k=best_k)
        knn_best.fit(X_tr, y_tr)
        y_pred = knn_best.predict(X_te)

        # metric comparison
        metric_cmp = compare_metrics(X_tr, y_tr, X_te, y_te, k=best_k)

        # weight comparison
        wk_vals, weight_cmp = compare_weights(X_tr, y_tr, X_te, y_te)

        all_results[name] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "target_names": target_names,
            "k_vals": k_vals,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "best_k": best_k,
            "cv_k_vals": cv_k_vals,
            "cv_mean_accs": cv_mean_accs,
            "cv_best_k": cv_best_k,
            "y_pred": y_pred,
            "metric_cmp": metric_cmp,
            "wk_vals": wk_vals,
            "weight_cmp": weight_cmp,
        }

    return all_results
