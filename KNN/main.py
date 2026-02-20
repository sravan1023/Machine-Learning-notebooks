import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import run_all
from metrics import confusion_matrix, classification_report

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Plot helpers ─────────────────────────────────────────────────────
def _save(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_k_accuracy(k_vals, train_accs, test_accs, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, train_accs, "bo-", linewidth=2, markersize=5, label="Train")
    ax.plot(k_vals, test_accs, "ro-", linewidth=2, markersize=5, label="Test")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_k_accuracy.png")


def plot_cv_accuracy(cv_k_vals, cv_mean_accs, cv_best_k, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cv_k_vals, cv_mean_accs, "go-", linewidth=2, markersize=5)
    ax.axvline(cv_best_k, color="red", linestyle="--", label=f"Best k={cv_best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title(f"{name} – Cross-Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_cv_accuracy.png")


def plot_confusion_matrix(y_true, y_pred, target_names, name):
    cm, labels = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    # annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} – Confusion Matrix")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_confusion.png")


def plot_metric_comparison(metric_cmp, name):
    metrics = list(metric_cmp.keys())
    accs = [metric_cmp[m]["accuracy"] for m in metrics]
    f1s = [metric_cmp[m]["macro_f1"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(f"{name} – Distance Metric Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_metric_cmp.png")


def plot_weight_comparison(wk_vals, weight_cmp, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    for w, accs in weight_cmp.items():
        ax.plot(wk_vals, accs, "o-", linewidth=2, markersize=5, label=w)
    ax.set_xlabel("k")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"{name} – Uniform vs Distance Weighting")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_weights.png")


def plot_decision_boundary(X_train, y_train, k, name, metric="euclidean"):
    """2-D decision boundary using the first two features."""
    from knn import KNNClassifier

    X2 = X_train[:, :2]
    knn = KNNClassifier(k=k, metric=metric)
    knn.fit(X2, y_train)

    h = 0.15
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y_train,
                         cmap="viridis", edgecolors="k", s=30)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{name} – Decision Boundary (k={k}, first 2 features)")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_boundary.png")


# ── Console report ───────────────────────────────────────────────────
def print_results(all_results):
    for name, res in all_results.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"Best k (test):  {res['best_k']}")
        print(f"Best k (CV):    {res['cv_best_k']}")
        print(f"Test accuracy:  {res['test_accs'][res['k_vals'].index(res['best_k'])]:.4f}")

        print(f"\nDistance-metric comparison (k={res['best_k']}):")
        for m, v in res["metric_cmp"].items():
            print(f"  {m:12s}  acc={v['accuracy']:.4f}  macro-f1={v['macro_f1']:.4f}")

        print(f"\nClassification report (k={res['best_k']}):")
        print(classification_report(res["y_test"], res["y_pred"],
                                    label_names=res["target_names"]))


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        plot_k_accuracy(res["k_vals"], res["train_accs"],
                        res["test_accs"], name)
        plot_cv_accuracy(res["cv_k_vals"], res["cv_mean_accs"],
                         res["cv_best_k"], name)
        plot_confusion_matrix(res["y_test"], res["y_pred"],
                              res["target_names"], name)
        plot_metric_comparison(res["metric_cmp"], name)
        plot_weight_comparison(res["wk_vals"], res["weight_cmp"], name)
        plot_decision_boundary(res["X_train"], res["y_train"],
                               res["best_k"], name)

    print_results(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
