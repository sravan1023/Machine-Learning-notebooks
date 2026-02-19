import numpy as np


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


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

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    return {
        "per_class": {
            "labels": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
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
