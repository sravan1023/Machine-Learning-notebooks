import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import run_all, accuracy, confusion_matrix, classification_report

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Plot helpers ─────────────────────────────────────────────────────
def _save(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_history, name, label=""):
    """Plot training loss vs iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # multi-class: loss_history is a list of lists (one per class)
    if isinstance(loss_history[0], list):
        for i, hist in enumerate(loss_history):
            ax.plot(hist, linewidth=1.5, label=f"class {i}")
        ax.legend()
    else:
        ax.plot(loss_history, linewidth=2, color="steelblue",
                label=label or "loss")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title(f"{name} – Training Loss Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_loss_curve.png")


def plot_lr_comparison(lr_sweep, name):
    """Plot loss curves for different learning rates."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for lr, info in sorted(lr_sweep.items()):
        hist = info["loss_history"]
        # for multi-class, average the per-class losses
        if isinstance(hist[0], list):
            max_len = max(len(h) for h in hist)
            avg = np.zeros(max_len)
            for h in hist:
                avg[:len(h)] += np.array(h)
            avg /= len(hist)
            ax.plot(avg, linewidth=1.5, label=f"lr={lr}")
        else:
            ax.plot(hist, linewidth=1.5, label=f"lr={lr}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} – Loss vs Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_lr_comparison.png")


def plot_lr_accuracy(lr_sweep, name):
    """Bar chart of test accuracy per learning rate."""
    lrs = sorted(lr_sweep.keys())
    test_accs = [lr_sweep[lr]["test_acc"] for lr in lrs]
    train_accs = [lr_sweep[lr]["train_acc"] for lr in lrs]

    x = np.arange(len(lrs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, train_accs, width, label="Train")
    ax.bar(x + width / 2, test_accs, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels([str(lr) for lr in lrs])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs Learning Rate")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_lr_accuracy.png")


def plot_regularization(reg_sweep, name):
    """Train/test accuracy vs regularization strength C."""
    cs = sorted(reg_sweep.keys())
    train_accs = [reg_sweep[c]["train_acc"] for c in cs]
    test_accs = [reg_sweep[c]["test_acc"] for c in cs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(cs, train_accs, "bo-", linewidth=2, markersize=6, label="Train")
    ax.semilogx(cs, test_accs, "ro-", linewidth=2, markersize=6, label="Test")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs Regularization (L2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_regularization.png")


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


def plot_decision_boundary(X_train, y_train, model, name):
    """2-D decision boundary using the first two features."""
    from model import LogisticRegression

    X2 = X_train[:, :2]
    lr2 = LogisticRegression(lr=model.lr, max_iter=model.max_iter,
                             regularization=model.regularization,
                             C=model.C, random_state=42)
    lr2.fit(X2, y_train)

    h = 0.05
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = lr2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X2[:, 0], X2[:, 1], c=y_train,
               cmap="viridis", edgecolors="k", s=30)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{name} – Decision Boundary (first 2 features)")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_boundary.png")


def plot_cv_accuracy(cv_accs, name):
    """Bar chart of per-fold cross-validation accuracy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = np.arange(1, len(cv_accs) + 1)
    ax.bar(folds, cv_accs, color="steelblue", edgecolor="black")
    ax.axhline(cv_accs.mean(), color="red", linestyle="--",
               label=f"Mean = {cv_accs.mean():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Cross-Validation Accuracy")
    ax.set_xticks(folds)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_cv_accuracy.png")


# ── Console report ───────────────────────────────────────────────────
def print_results(all_results):
    for name, res in all_results.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"Best learning rate:  {res['best_lr']}")
        print(f"Best C (L2):         {res['best_C']}")
        print(f"Test accuracy:       {accuracy(res['y_test'], res['y_pred']):.4f}")
        print(f"CV accuracy:         {res['cv_accs'].mean():.4f} "
              f"(± {res['cv_accs'].std():.4f})")

        print(f"\nLearning-rate sweep:")
        for lr in sorted(res["lr_sweep"]):
            info = res["lr_sweep"][lr]
            print(f"  lr={lr:<6}  train={info['train_acc']:.4f}  "
                  f"test={info['test_acc']:.4f}")

        print(f"\nRegularization sweep:")
        for c in sorted(res["reg_sweep"]):
            info = res["reg_sweep"][c]
            print(f"  C={c:<8}  train={info['train_acc']:.4f}  "
                  f"test={info['test_acc']:.4f}")

        print(f"\nClassification report:")
        print(classification_report(res["y_test"], res["y_pred"],
                                    label_names=res["target_names"]))


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        plot_loss_curve(res["best_model"].loss_history_, name)
        plot_lr_comparison(res["lr_sweep"], name)
        plot_lr_accuracy(res["lr_sweep"], name)
        plot_regularization(res["reg_sweep"], name)
        plot_confusion_matrix(res["y_test"], res["y_pred"],
                              res["target_names"], name)
        plot_decision_boundary(res["X_train"], res["y_train"],
                               res["best_model"], name)
        plot_cv_accuracy(res["cv_accs"], name)

    print_results(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
