import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from network import NeuralNetwork
from optimizer import BatchGradientDescent


# Helper utilities
def one_hot(y, n_classes):
    """Convert integer labels to one-hot encoding.

    Parameters
    ----------
    y : ndarray of shape (m,)
        Integer class labels.
    n_classes : int
        Total number of classes.

    Returns
    -------
    ndarray of shape (m, n_classes)
    """
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


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


# ── 1  XOR experiment (non-linear toy problem) ──────────────────────
def run_xor(lr=2.0, epochs=10000, hidden_size=8, random_state=42):
    """Train a 1-hidden-layer MLP on XOR.

    Architecture: 2 -> hidden_size (ReLU) -> 1 (Sigmoid)
    Loss:         binary cross-entropy

    Returns
    -------
    dict with keys: model, X, y, y_pred, loss_history
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = NeuralNetwork(
        layer_sizes=[2, hidden_size, 1],
        activations=["relu", "sigmoid"],
        loss="binary_cross_entropy",
        optimizer=BatchGradientDescent(lr=lr),
        random_state=random_state,
    )
    model.fit(X, y, epochs=epochs, verbose=True, print_every=epochs // 10)

    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    print("\nXOR predictions:")
    for xi, pi, yi in zip(X, proba, y.ravel()):
        print(f"  {xi}  →  prob={pi[0]:.4f}  pred={int(pi[0]>=0.5)}  "
              f"true={int(yi)}")
    print(f"  Accuracy: {accuracy(y.ravel(), y_pred):.4f}")

    return {
        "model": model,
        "X": X, "y": y,
        "y_pred": y_pred,
        "y_proba": proba,
        "loss_history": model.loss_history_,
    }


# 2  Digits dataset experiment
def run_digits(lr=0.1, epochs=500, hidden_size=64,
               test_size=0.3, random_state=42):
    """Train a 1-hidden-layer MLP on sklearn's 8×8 digits.

    Architecture: 64 -> hidden_size (ReLU) -> 10 (Softmax)
    Loss:         categorical cross-entropy

    Returns
    -------
    dict with keys: model, X_train, X_test, y_train, y_test,
                    y_pred, loss_history, target_names
    """
    data = load_digits()
    X = StandardScaler().fit_transform(data.data.astype(float))
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    n_classes = len(np.unique(y))
    y_train_oh = one_hot(y_train, n_classes)

    model = NeuralNetwork(
        layer_sizes=[X.shape[1], hidden_size, n_classes],
        activations=["relu", "softmax"],
        loss="categorical_cross_entropy",
        optimizer=BatchGradientDescent(lr=lr),
        random_state=random_state,
    )
    model.fit(X_train, y_train_oh, epochs=epochs, verbose=True,
              print_every=epochs // 10)

    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train_oh)
    test_acc = accuracy(y_test, y_pred)

    target_names = [str(i) for i in range(n_classes)]

    print(f"\nDigits – Train accuracy: {train_acc:.4f}")
    print(f"Digits – Test  accuracy: {test_acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, label_names=target_names)}")

    return {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "loss_history": model.loss_history_,
        "target_names": target_names,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


# 3  Learning-rate sweep on Digits
def sweep_lr_digits(lr_values=(0.001, 0.01, 0.05, 0.1, 0.5),
                    epochs=300, hidden_size=64,
                    test_size=0.3, random_state=42):
    """Train Digits at different learning rates; return results dict."""
    data = load_digits()
    X = StandardScaler().fit_transform(data.data.astype(float))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    n_classes = len(np.unique(y))
    y_train_oh = one_hot(y_train, n_classes)

    results = {}
    for lr in lr_values:
        model = NeuralNetwork(
            layer_sizes=[X.shape[1], hidden_size, n_classes],
            activations=["relu", "softmax"],
            loss="categorical_cross_entropy",
            optimizer=BatchGradientDescent(lr=lr),
            random_state=random_state,
        )
        model.fit(X_train, y_train_oh, epochs=epochs, verbose=False)

        train_acc = model.score(X_train, y_train_oh)
        test_acc = accuracy(y_test, model.predict(X_test))

        results[lr] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "loss_history": model.loss_history_,
        }
        print(f"  lr={lr:<6}  train={train_acc:.4f}  test={test_acc:.4f}")

    return results


# 4  Hidden-size sweep on Digits
def sweep_hidden_size(hidden_sizes=(8, 16, 32, 64, 128),
                      lr=0.1, epochs=300,
                      test_size=0.3, random_state=42):
    """Train Digits at different hidden layer sizes."""
    data = load_digits()
    X = StandardScaler().fit_transform(data.data.astype(float))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    n_classes = len(np.unique(y))
    y_train_oh = one_hot(y_train, n_classes)

    results = {}
    for h in hidden_sizes:
        model = NeuralNetwork(
            layer_sizes=[X.shape[1], h, n_classes],
            activations=["relu", "softmax"],
            loss="categorical_cross_entropy",
            optimizer=BatchGradientDescent(lr=lr),
            random_state=random_state,
        )
        model.fit(X_train, y_train_oh, epochs=epochs, verbose=False)

        train_acc = model.score(X_train, y_train_oh)
        test_acc = accuracy(y_test, model.predict(X_test))

        results[h] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "loss_history": model.loss_history_,
        }
        print(f"  hidden={h:<4}  train={train_acc:.4f}  test={test_acc:.4f}")

    return results


def run_all():
    """Execute all experiments and return a results dictionary."""
    print("=" * 60)
    print("  Experiment 1: XOR (non-linear)")
    print("=" * 60)
    xor_results = run_xor()

    print(f"\n{'=' * 60}")
    print("  Experiment 2: Digits (8×8 hand-written)")
    print("=" * 60)
    digits_results = run_digits()

    print(f"\n{'=' * 60}")
    print("  Experiment 3: Learning-rate sweep (Digits)")
    print("=" * 60)
    lr_sweep = sweep_lr_digits()

    print(f"\n{'=' * 60}")
    print("  Experiment 4: Hidden-size sweep (Digits)")
    print("=" * 60)
    hidden_sweep = sweep_hidden_size()

    return {
        "xor": xor_results,
        "digits": digits_results,
        "lr_sweep": lr_sweep,
        "hidden_sweep": hidden_sweep,
    }
