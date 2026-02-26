import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import (run_all, accuracy, confusion_matrix,
                         classification_report)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# Plot helpers
def _save(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_depth_accuracy(depth_vals, train_accs, test_accs, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_vals, train_accs, "bo-", linewidth=2, markersize=5, label="Train")
    ax.plot(depth_vals, test_accs, "ro-", linewidth=2, markersize=5, label="Test")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs Tree Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_depth_accuracy.png")


def plot_cv_depth(cv_depths, cv_mean_accs, cv_best_depth, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cv_depths, cv_mean_accs, "go-", linewidth=2, markersize=5)
    ax.axvline(cv_best_depth, color="red", linestyle="--", label=f"Best depth={cv_best_depth}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title(f"{name} – Cross-Validation Accuracy vs Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_cv_depth.png")


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


def plot_criteria_comparison(crit_cmp, name):
    criteria = list(crit_cmp.keys())
    accs = [crit_cmp[c]["accuracy"] for c in criteria]
    f1s = [crit_cmp[c]["macro_f1"] for c in criteria]

    x = np.arange(len(criteria))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria)
    ax.set_ylabel("Score")
    ax.set_title(f"{name} – Criterion Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_criteria_cmp.png")


def plot_feature_importance(importances, feature_names, name, top_n=10):
    """Horizontal bar chart of the top-N most important features."""
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color="steelblue", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{name} – Feature Importance (top {len(indices)})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_feature_importance.png")


def plot_min_samples_split(mss_sweep, name):
    mss_vals = sorted(mss_sweep.keys())
    train_accs = [mss_sweep[m]["train_acc"] for m in mss_vals]
    test_accs = [mss_sweep[m]["test_acc"] for m in mss_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mss_vals, train_accs, "bo-", linewidth=2, markersize=6, label="Train")
    ax.plot(mss_vals, test_accs, "ro-", linewidth=2, markersize=6, label="Test")
    ax.set_xlabel("min_samples_split")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs min_samples_split")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_min_samples_split.png")


def plot_decision_boundary(X_train, y_train, model, name):
    """2-D decision boundary using the first two features."""
    from tree import DecisionTreeClassifier

    X2 = X_train[:, :2]
    dt2 = DecisionTreeClassifier(criterion=model.criterion,
                                 max_depth=model.max_depth,
                                 min_samples_split=model.min_samples_split,
                                 min_samples_leaf=model.min_samples_leaf)
    dt2.fit(X2, y_train)

    h = 0.05
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = dt2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X2[:, 0], X2[:, 1], c=y_train, cmap="viridis", edgecolors="k", s=30)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{name} – Decision Boundary (first 2 features)")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_boundary.png")


def print_results(all_results):
    for name, res in all_results.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"Best depth (test):       {res['best_depth']}")
        print(f"Best depth (CV):         {res['cv_best_depth']}")
        print(f"Test accuracy:           "
            f"{accuracy(res['y_test'], res['y_pred']):.4f}")
        print(f"Tree depth (best):       {res['best_model'].get_depth()}")
        print(f"Leaves (best):           {res['best_model'].get_n_leaves()}")
        print(f"Full-tree depth:         {res['full_model'].get_depth()}")
        print(f"Full-tree leaves:        {res['full_model'].get_n_leaves()}")

        print(f"\nCriterion comparison:")
        for c, v in res["crit_cmp"].items():
            print(f"  {c:20s}  acc={v['accuracy']:.4f}  "
                  f"macro-f1={v['macro_f1']:.4f}  "
                  f"depth={v['depth']}  leaves={v['n_leaves']}")

        print(f"\nmin_samples_split sweep:")
        for mss in sorted(res["mss_sweep"]):
            v = res["mss_sweep"][mss]
            print(f"  mss={mss:<4}  train={v['train_acc']:.4f}  "
                  f"test={v['test_acc']:.4f}  "
                  f"depth={v['depth']}  leaves={v['n_leaves']}")

        print(f"\nClassification report (depth={res['cv_best_depth']}):")
        print(classification_report(res["y_test"], res["y_pred"], label_names=res["target_names"]))


# Entry point 
if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        plot_depth_accuracy(res["depth_vals"], res["train_accs"],  res["test_accs"], name)
        plot_cv_depth(res["cv_depths"], res["cv_mean_accs"], res["cv_best_depth"], name)
        plot_confusion_matrix(res["y_test"], res["y_pred"], res["target_names"], name)
        plot_criteria_comparison(res["crit_cmp"], name)
        plot_feature_importance(res["best_model"].feature_importances_, res["feature_names"], name)
        plot_min_samples_split(res["mss_sweep"], name)
        plot_decision_boundary(res["X_train"], res["y_train"], res["best_model"], name)

    print_results(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
