import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import (run_all, accuracy, confusion_matrix, classification_report)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# Plot helpers
def _save(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _slug(name):
    return name.lower().replace(" ", "_")


# n_estimators vs accuracy
def plot_n_estimators(n_vals, train_accs, test_accs, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_vals, train_accs, "bo-", lw=2, ms=5, label="Train")
    ax.plot(n_vals, test_accs, "ro-", lw=2, ms=5, label="Test")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs Number of Trees")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_n_estimators.png")


# max_depth vs accuracy
def plot_depth_accuracy(depth_vals, train_accs, test_accs, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_vals, train_accs, "bo-", lw=2, ms=5, label="Train")
    ax.plot(depth_vals, test_accs, "ro-", lw=2, ms=5, label="Test")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Random Forest Accuracy vs Tree Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_depth_accuracy.png")


# OOB error curve
def plot_oob_error(oob_ns, oob_errors, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(oob_ns, oob_errors, "go-", lw=2, ms=6)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("OOB Error")
    ax.set_title(f"{name} – Out-of-Bag Error vs Number of Trees")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_oob_error.png")


# max_features comparison
def plot_max_features(mf_results, name):
    labels = list(mf_results.keys())
    train_accs = [mf_results[l]["train_acc"] for l in labels]
    test_accs = [mf_results[l]["test_acc"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, train_accs, width, label="Train", color="steelblue")
    ax.bar(x + width / 2, test_accs, width, label="Test", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – max_features Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_max_features.png")


# RF vs Decision Tree
def plot_rf_vs_tree(vs_tree, name):
    labels = ["Decision Tree", "Random Forest"]
    train = [vs_tree["dt_train"], vs_tree["rf_train"]]
    test = [vs_tree["dt_test"], vs_tree["rf_test"]]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_train = ax.bar(x - width / 2, train, width,
                        label="Train", color="steelblue")
    bars_test = ax.bar(x + width / 2, test, width,
                       label="Test", color="salmon")

    # annotate OOB on the RF test bar
    oob = vs_tree.get("rf_oob")
    if oob is not None:
        ax.annotate(f"OOB={oob:.3f}", xy=(1 + width / 2, test[1]),
                    xytext=(1 + width / 2 + 0.15, test[1] - 0.05),
                    fontsize=9, arrowprops=dict(arrowstyle="->"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Random Forest vs Single Tree")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_rf_vs_tree.png")


# Confusion matrix
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
    ax.set_title(f"{name} – Confusion Matrix (Random Forest)")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_confusion.png")


# Feature importance
def plot_feature_importance(importances, feature_names, name, top_n=10):
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color="forestgreen", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{name} – Feature Importance (Random Forest, top {len(indices)})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_feature_importance.png")


# CV n_estimators
def plot_cv_n_estimators(cv_ns, cv_accs, cv_best_n, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cv_ns, cv_accs, "mo-", lw=2, ms=6)
    ax.axvline(cv_best_n, color="red", ls="--",  label=f"Best n_estimators={cv_best_n}")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title(f"{name} – CV Accuracy vs n_estimators")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_cv_n_estimators.png")


# Decision boundary (first 2 features)
def plot_decision_boundary(X_train, y_train, model, name):
    from forest import RandomForestClassifier

    X2 = X_train[:, :2]
    rf2 = RandomForestClassifier(
        n_estimators=model.n_estimators,
        max_depth=model.max_depth,
        random_state=42,
    )
    rf2.fit(X2, y_train)

    h = 0.05
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = rf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X2[:, 0], X2[:, 1], c=y_train, cmap="viridis", edgecolors="k", s=30)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{name} – Decision Boundary (Random Forest, first 2 features)")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_boundary.png")


# Console report
def print_results(all_results):
    for name, res in all_results.items():
        print()
        print(f"  {name}")
        print()
        print(f"Best n_estimators (CV):  {res['cv_best_n']}")
        print(f"Best max_depth (test):   {res['best_depth']}")
        print(f"Test accuracy:           "
              f"{accuracy(res['y_test'], res['y_pred']):.4f}")
        oob = res['best_rf'].oob_score_
        if oob is not None:
            print(f"OOB accuracy:            {oob:.4f}")

        print(f"\nRF vs Single Decision Tree:")
        vs = res["vs_tree"]
        print(f"  DT  train={vs['dt_train']:.4f}  test={vs['dt_test']:.4f}")
        print(f"  RF  train={vs['rf_train']:.4f}  test={vs['rf_test']:.4f}"
              f"  oob={vs['rf_oob']:.4f}")

        print(f"\nmax_features comparison:")
        for label, v in res["mf_results"].items():
            print(f"  {label:>6s}  train={v['train_acc']:.4f}  "
                  f"test={v['test_acc']:.4f}")

        print(f"\nClassification report:")
        print(classification_report(res["y_test"], res["y_pred"],
                                    label_names=res["target_names"]))


# Entry point
if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        plot_n_estimators(res["n_vals"], res["n_train_accs"], res["n_test_accs"], name)
        plot_depth_accuracy(res["depth_vals"], res["d_train_accs"], res["d_test_accs"], name)
        plot_oob_error(res["oob_ns"], res["oob_errors"], name)
        plot_max_features(res["mf_results"], name)
        plot_rf_vs_tree(res["vs_tree"], name)
        plot_confusion_matrix(res["y_test"], res["y_pred"], res["target_names"], name)
        plot_feature_importance(res["best_rf"].feature_importances_, res["feature_names"], name)
        plot_cv_n_estimators(res["cv_ns"], res["cv_accs"], res["cv_best_n"], name)
        plot_decision_boundary(res["X_train"], res["y_train"], res["best_rf"], name)

    print_results(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
