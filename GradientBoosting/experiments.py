import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from boosting import GradientBoostingClassifier


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
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "target_names": [str(t) for t in data.target_names],
        }
    return datasets


def sweep_n_estimators(X_train, y_train, X_test, y_test,
                       n_values=(5, 10, 20, 40),
                       learning_rate=0.1, max_depth=2, random_state=42):
    train_accs, test_accs = [], []
    for n in n_values:
        gb = GradientBoostingClassifier(
            n_estimators=n,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        gb.fit(X_train, y_train)
        train_accs.append(gb.score(X_train, y_train))
        test_accs.append(gb.score(X_test, y_test))
    return list(n_values), train_accs, test_accs


def sweep_learning_rate(X_train, y_train, X_test, y_test,
                        lr_values=(0.01, 0.05, 0.1, 0.2, 0.5),
                        n_estimators=40, max_depth=2, random_state=42):
    results = {}
    for lr in lr_values:
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            random_state=random_state,
        )
        gb.fit(X_train, y_train)
        results[lr] = {
            "train_acc": gb.score(X_train, y_train),
            "test_acc": gb.score(X_test, y_test),
            "loss_history": gb.loss_history_,
        }
    return results


def sweep_depth(X_train, y_train, X_test, y_test,
                depth_values=(1, 2, 3),
                n_estimators=40, learning_rate=0.1, random_state=42):
    train_accs, test_accs = [], []
    for d in depth_values:
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=d,
            random_state=random_state,
        )
        gb.fit(X_train, y_train)
        train_accs.append(gb.score(X_train, y_train))
        test_accs.append(gb.score(X_test, y_test))
    return list(depth_values), train_accs, test_accs


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

        n_vals, n_train, n_test = sweep_n_estimators(X_tr, y_tr, X_te, y_te)
        d_vals, d_train, d_test = sweep_depth(X_tr, y_tr, X_te, y_te)
        lr_sweep = sweep_learning_rate(X_tr, y_tr, X_te, y_te)

        best_n = n_vals[int(np.argmax(n_test))]
        best_depth = d_vals[int(np.argmax(d_test))]
        best_lr = max(lr_sweep, key=lambda k: lr_sweep[k]["test_acc"])

        best_model = GradientBoostingClassifier(
            n_estimators=best_n,
            learning_rate=best_lr,
            max_depth=best_depth,
            random_state=42,
        )
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_te)

        all_results[name] = {
            "X_train": X_tr,
            "X_test": X_te,
            "y_train": y_tr,
            "y_test": y_te,
            "target_names": d["target_names"],
            "n_vals": n_vals,
            "n_train": n_train,
            "n_test": n_test,
            "d_vals": d_vals,
            "d_train": d_train,
            "d_test": d_test,
            "lr_sweep": lr_sweep,
            "best_n": best_n,
            "best_depth": best_depth,
            "best_lr": best_lr,
            "best_model": best_model,
            "y_pred": y_pred,
        }

    return all_results
