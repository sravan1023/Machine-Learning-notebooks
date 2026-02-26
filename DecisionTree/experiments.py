import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tree import DecisionTreeClassifier


# Dataset helpers
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
            "feature_names": list(data.feature_names),
        }
    return datasets


# Sweep max_depth
def sweep_depth(X_train, y_train, X_test, y_test,
                depth_range=range(1, 21), criterion="gini"):
    """Train trees at increasing max_depth; return train/test accuracy."""
    train_accs, test_accs = [], []
    for d in depth_range:
        dt = DecisionTreeClassifier(criterion=criterion, max_depth=d)
        dt.fit(X_train, y_train)
        train_accs.append(dt.score(X_train, y_train))
        test_accs.append(dt.score(X_test, y_test))
    return list(depth_range), train_accs, test_accs


# Sweep min_samples_split 
def sweep_min_samples_split(X_train, y_train, X_test, y_test,
                            values=(2, 5, 10, 20, 50), criterion="gini"):
    results = {}
    for mss in values:
        dt = DecisionTreeClassifier(criterion=criterion,
                                    min_samples_split=mss)
        dt.fit(X_train, y_train)
        results[mss] = {
            "train_acc": dt.score(X_train, y_train),
            "test_acc": dt.score(X_test, y_test),
            "depth": dt.get_depth(),
            "n_leaves": dt.get_n_leaves(),
        }
    return results


# Compare criteria
def compare_criteria(X_train, y_train, X_test, y_test,
                     criteria=("gini", "entropy", "misclassification"),
                     max_depth=None):
    results = {}
    for c in criteria:
        dt = DecisionTreeClassifier(criterion=c, max_depth=max_depth)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_test)
        acc = float(np.mean(y_test == preds))
        prf = precision_recall_f1(y_test, preds)
        results[c] = {
            "accuracy": acc,
            "macro_f1": prf["macro"]["f1"],
            "depth": dt.get_depth(),
            "n_leaves": dt.get_n_leaves(),
        }
    return results


# Cross-validation (manual k-fold) 
def cross_validate(X, y, max_depth=None, criterion="gini",
                   n_folds=5, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_accs = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        dt.fit(X[train_idx], y[train_idx])
        fold_accs.append(dt.score(X[val_idx], y[val_idx]))

    return np.array(fold_accs)


#  Best depth via cross-validation
def best_depth_cv(X, y, depth_range=range(1, 21), n_folds=5,
                  criterion="gini", random_state=42):
    mean_accs = []
    for d in depth_range:
        accs = cross_validate(X, y, max_depth=d, criterion=criterion,
                              n_folds=n_folds, random_state=random_state)
        mean_accs.append(accs.mean())
    best_idx = int(np.argmax(mean_accs))
    return list(depth_range), mean_accs, list(depth_range)[best_idx]


# Metrics 
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def precision_recall_f1(y_true, y_pred, labels=None):
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    n = len(labels)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i] = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                 if (precision[i] + recall[i]) > 0 else 0.0)

    return {
        "per_class": {"labels": labels, "precision": precision,
                      "recall": recall, "f1": f1},
        "macro": {"precision": float(np.mean(precision)),
                  "recall": float(np.mean(recall)),
                  "f1": float(np.mean(f1))},
    }


def classification_report(y_true, y_pred, label_names=None):
    results = precision_recall_f1(y_true, y_pred)
    labels = results["per_class"]["labels"]
    prec = results["per_class"]["precision"]
    rec = results["per_class"]["recall"]
    f1 = results["per_class"]["f1"]

    header = f"{'Class':>15s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}"
    lines = [header, "-" * len(header)]

    for i, lbl in enumerate(labels):
        name = label_names[i] if label_names else str(lbl)
        lines.append(f"{name:>15s}  {prec[i]:9.4f}  {rec[i]:9.4f}  {f1[i]:9.4f}")

    lines.append("-" * len(header))
    m = results["macro"]
    lines.append(f"{'Macro avg':>15s}  {m['precision']:9.4f}  {m['recall']:9.4f}  {m['f1']:9.4f}")
    acc = accuracy(y_true, y_pred)
    lines.append(f"{'Accuracy':>15s}  {acc:9.4f}")

    return "\n".join(lines)


# Run everything 
def run_all():
    datasets = load_datasets()
    all_results = {}

    for name, d in datasets.items():
        X_tr, X_te = d["X_train"], d["X_test"]
        y_tr, y_te = d["y_train"], d["y_test"]
        target_names = d["target_names"]
        feature_names = d["feature_names"]

        # depth sweep
        depth_vals, train_accs, test_accs = sweep_depth(X_tr, y_tr, X_te, y_te)
        best_d_idx = int(np.argmax(test_accs))
        best_depth = depth_vals[best_d_idx]

        # cross-validation for best depth
        cv_depths, cv_mean_accs, cv_best_depth = best_depth_cv(
            np.vstack([X_tr, X_te]),
            np.concatenate([y_tr, y_te]))

        # min_samples_split sweep
        mss_sweep = sweep_min_samples_split(X_tr, y_tr, X_te, y_te)

        # criterion comparison
        crit_cmp = compare_criteria(X_tr, y_tr, X_te, y_te)

        # best model (use CV best depth)
        best_model = DecisionTreeClassifier(criterion="gini",
                                            max_depth=cv_best_depth)
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_te)

        # fully grown tree (no depth limit) for comparison
        full_model = DecisionTreeClassifier(criterion="gini")
        full_model.fit(X_tr, y_tr)

        all_results[name] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "target_names": target_names,
            "feature_names": feature_names,
            "depth_vals": depth_vals,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "best_depth": best_depth,
            "cv_depths": cv_depths,
            "cv_mean_accs": cv_mean_accs,
            "cv_best_depth": cv_best_depth,
            "mss_sweep": mss_sweep,
            "crit_cmp": crit_cmp,
            "best_model": best_model,
            "full_model": full_model,
            "y_pred": y_pred,
        }

    return all_results
