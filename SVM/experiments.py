import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from svm import SVC


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


# Compare kernels
def compare_kernels(X_train, y_train, X_test, y_test,
                    kernels=("linear", "poly", "rbf", "sigmoid"), C=1.0, random_state=42):
    
    """Train an SVM with each kernel and collect accuracy + #SVs."""
    
    results = {}
    for k in kernels:
        model = SVC(C=C, kernel=k, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prf = precision_recall_f1(y_test, y_pred)
        results[k] = {
            "train_acc": model.score(X_train, y_train),
            "test_acc": model.score(X_test, y_test),
            "macro_f1": prf["macro"]["f1"],
            "n_support": model.get_n_support(),
        }
    return results


# Sweep C (regularization)
def sweep_C(X_train, y_train, X_test, y_test,
            C_values=(0.01, 0.1, 1.0, 10.0, 100.0),
            kernel="rbf", random_state=42):
    results = {}
    for C in C_values:
        model = SVC(C=C, kernel=kernel, random_state=random_state)
        model.fit(X_train, y_train)
        results[C] = {
            "train_acc": model.score(X_train, y_train),
            "test_acc": model.score(X_test, y_test),
            "n_support": model.get_n_support(),
        }
    return results


# Sweep gamma (RBF)
def sweep_gamma(X_train, y_train, X_test, y_test,
                gamma_values=(0.001, 0.01, 0.1, 1.0, 10.0),
                C=1.0, random_state=42):
    results = {}
    for g in gamma_values:
        model = SVC(C=C, kernel="rbf", gamma=g, random_state=random_state)
        model.fit(X_train, y_train)
        results[g] = {
            "train_acc": model.score(X_train, y_train),
            "test_acc": model.score(X_test, y_test),
            "n_support": model.get_n_support(),
        }
    return results


# Sweep polynomial degree
def sweep_degree(X_train, y_train, X_test, y_test,
                 degrees=(2, 3, 4, 5), C=1.0, random_state=42):
    results = {}
    for d in degrees:
        model = SVC(C=C, kernel="poly", degree=d, random_state=random_state)
        model.fit(X_train, y_train)
        results[d] = {
            "train_acc": model.score(X_train, y_train),
            "test_acc": model.score(X_test, y_test),
            "n_support": model.get_n_support(),
        }
    return results


# Cross-validation
def cross_validate(X, y, kernel="rbf", C=1.0, n_folds=5, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_accs = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        model = SVC(C=C, kernel=kernel, random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        fold_accs.append(model.score(X[val_idx], y[val_idx]))

    return np.array(fold_accs)


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
    lines.append(f"{'Macro avg':>15s}  {m['precision']:9.4f}  "
                 f"{m['recall']:9.4f}  {m['f1']:9.4f}")
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

        print(f"[{name}] comparing kernels …")
        kernel_cmp = compare_kernels(X_tr, y_tr, X_te, y_te)
        best_kernel = max(kernel_cmp,
                          key=lambda k: kernel_cmp[k]["test_acc"])

        print(f"[{name}] sweeping C …")
        c_sweep = sweep_C(X_tr, y_tr, X_te, y_te, kernel=best_kernel)
        best_C = max(c_sweep, key=lambda k: c_sweep[k]["test_acc"])

        print(f"[{name}] sweeping gamma …")
        gamma_sweep = sweep_gamma(X_tr, y_tr, X_te, y_te, C=best_C)
        best_gamma = max(gamma_sweep,
                         key=lambda k: gamma_sweep[k]["test_acc"])

        print(f"[{name}] sweeping poly degree …")
        degree_sweep = sweep_degree(X_tr, y_tr, X_te, y_te, C=best_C)

        print(f"[{name}] cross-validating …")
        cv_accs = cross_validate(
            np.vstack([X_tr, X_te]), np.concatenate([y_tr, y_te]),
            kernel=best_kernel, C=best_C)

        # best model
        best_model = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma,
                         random_state=42)
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_te)

        all_results[name] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "target_names": target_names,
            "feature_names": feature_names,
            "kernel_cmp": kernel_cmp,
            "best_kernel": best_kernel,
            "c_sweep": c_sweep,
            "best_C": best_C,
            "gamma_sweep": gamma_sweep,
            "best_gamma": best_gamma,
            "degree_sweep": degree_sweep,
            "cv_accs": cv_accs,
            "best_model": best_model,
            "y_pred": y_pred,
        }

    return all_results
