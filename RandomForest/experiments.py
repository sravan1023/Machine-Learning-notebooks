import sys
import os
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DecisionTree"))

from forest import RandomForestClassifier
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


# Sweep n_estimators 
def sweep_n_estimators(X_train, y_train, X_test, y_test,
                       n_range=(1, 5, 10, 25, 50, 100, 200),
                       random_state=42):
    """Train forests with increasing n_estimators; return accuracies."""
    train_accs, test_accs = [], []
    for n in n_range:
        rf = RandomForestClassifier(n_estimators=n, random_state=random_state)
        rf.fit(X_train, y_train)
        train_accs.append(rf.score(X_train, y_train))
        test_accs.append(rf.score(X_test, y_test))
    return list(n_range), train_accs, test_accs


# Sweep max_depth 
def sweep_depth(X_train, y_train, X_test, y_test,
                depth_range=range(1, 21), n_estimators=100,
                random_state=42):
    train_accs, test_accs = [], []
    for d in depth_range:
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=d,
                                    random_state=random_state)
        rf.fit(X_train, y_train)
        train_accs.append(rf.score(X_train, y_train))
        test_accs.append(rf.score(X_test, y_test))
    return list(depth_range), train_accs, test_accs


# Sweep max_features 
def sweep_max_features(X_train, y_train, X_test, y_test,
                       n_estimators=100, random_state=42):
    """Compare different max_features strategies."""
    n_feat = X_train.shape[1]
    strategies = {
        "sqrt": "sqrt",
        "log2": "log2",
        "all": None,
        "half": max(1, n_feat // 2),
    }
    results = {}
    for label, mf in strategies.items():
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_features=mf,
                                    random_state=random_state)
        rf.fit(X_train, y_train)
        results[label] = {
            "train_acc": rf.score(X_train, y_train),
            "test_acc": rf.score(X_test, y_test),
        }
    return results


# RF vs single Decision Tree
def rf_vs_single_tree(X_train, y_train, X_test, y_test,
                      n_estimators=100, max_depth=None, random_state=42):
    dt = DecisionTreeClassifier(criterion="gini", max_depth=max_depth)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state,
                                oob_score=True)
    rf.fit(X_train, y_train)

    return {
        "dt_train": dt.score(X_train, y_train),
        "dt_test": dt.score(X_test, y_test),
        "rf_train": rf.score(X_train, y_train),
        "rf_test": rf.score(X_test, y_test),
        "rf_oob": rf.oob_score_,
    }


# Cross-validation 
def cross_validate(X, y, n_estimators=100, max_depth=None,
                   n_folds=5, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_accs = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state)
        rf.fit(X[train_idx], y[train_idx])
        fold_accs.append(rf.score(X[val_idx], y[val_idx]))
    return np.array(fold_accs)


# Best n_estimators via CV
def best_n_estimators_cv(X, y, n_range=(10, 25, 50, 100, 200),
                         n_folds=5, random_state=42):
    mean_accs = []
    for n in n_range:
        accs = cross_validate(X, y, n_estimators=n,
                              n_folds=n_folds, random_state=random_state)
        mean_accs.append(accs.mean())
    best_idx = int(np.argmax(mean_accs))
    return list(n_range), mean_accs, list(n_range)[best_idx]


# OOB error curve
def oob_error_curve(X_train, y_train,
                    n_range=(1, 5, 10, 25, 50, 100, 200),
                    random_state=42):
    # OOB error (1 - OOB accuracy) as a function of n_estimators
    oob_errors = []
    for n in n_range:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                    random_state=random_state)
        rf.fit(X_train, y_train)
        oob_errors.append(1.0 - rf.oob_score_)
    return list(n_range), oob_errors


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

def run_all():
    datasets = load_datasets()
    all_results = {}

    for name, d in datasets.items():
        X_tr, X_te = d["X_train"], d["X_test"]
        y_tr, y_te = d["y_train"], d["y_test"]
        target_names = d["target_names"]
        feature_names = d["feature_names"]

        # n_estimators sweep
        n_vals, n_train_accs, n_test_accs = sweep_n_estimators(
            X_tr, y_tr, X_te, y_te)

        # depth sweep
        depth_vals, d_train_accs, d_test_accs = sweep_depth(
            X_tr, y_tr, X_te, y_te)
        best_d_idx = int(np.argmax(d_test_accs))
        best_depth = depth_vals[best_d_idx]

        # max_features comparison
        mf_results = sweep_max_features(X_tr, y_tr, X_te, y_te)

        # RF vs single tree
        vs_tree = rf_vs_single_tree(X_tr, y_tr, X_te, y_te)

        # OOB error curve
        oob_ns, oob_errors = oob_error_curve(X_tr, y_tr)

        # CV for best n_estimators
        cv_ns, cv_accs, cv_best_n = best_n_estimators_cv(
            np.vstack([X_tr, X_te]), np.concatenate([y_tr, y_te]))

        # final best model
        best_rf = RandomForestClassifier(n_estimators=cv_best_n,
                                         max_depth=best_depth,
                                         oob_score=True,
                                         random_state=42)
        best_rf.fit(X_tr, y_tr)
        y_pred = best_rf.predict(X_te)

        all_results[name] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "target_names": target_names,
            "feature_names": feature_names,
            # n_estimators sweep
            "n_vals": n_vals,
            "n_train_accs": n_train_accs,
            "n_test_accs": n_test_accs,
            # depth sweep
            "depth_vals": depth_vals,
            "d_train_accs": d_train_accs,
            "d_test_accs": d_test_accs,
            "best_depth": best_depth,
            # max_features
            "mf_results": mf_results,
            # RF vs tree
            "vs_tree": vs_tree,
            # OOB
            "oob_ns": oob_ns,
            "oob_errors": oob_errors,
            # CV
            "cv_ns": cv_ns,
            "cv_accs": cv_accs,
            "cv_best_n": cv_best_n,
            # best model
            "best_rf": best_rf,
            "y_pred": y_pred,
        }

    return all_results
