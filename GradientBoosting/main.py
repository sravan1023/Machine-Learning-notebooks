import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import run_all, accuracy, confusion_matrix, classification_report

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def _save(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _slug(name):
    return name.lower().replace(" ", "_")


def plot_n_estimators(n_vals, train_accs, test_accs, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_vals, train_accs, "bo-", linewidth=2, markersize=5, label="Train")
    ax.plot(n_vals, test_accs, "ro-", linewidth=2, markersize=5, label="Test")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} - Accuracy vs Number of Boosting Stages")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_n_estimators.png")


def plot_depth(d_vals, train_accs, test_accs, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d_vals, train_accs, "bo-", linewidth=2, markersize=5, label="Train")
    ax.plot(d_vals, test_accs, "ro-", linewidth=2, markersize=5, label="Test")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} - Accuracy vs Weak Learner Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_depth.png")


def plot_lr_accuracy(lr_sweep, name):
    lrs = sorted(lr_sweep.keys())
    train_accs = [lr_sweep[lr]["train_acc"] for lr in lrs]
    test_accs = [lr_sweep[lr]["test_acc"] for lr in lrs]

    x = np.arange(len(lrs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, train_accs, width, label="Train")
    ax.bar(x + width / 2, test_accs, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels([str(lr) for lr in lrs])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name} - Accuracy vs Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_lr_accuracy.png")


def plot_lr_loss(lr_sweep, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    for lr in sorted(lr_sweep.keys()):
        ax.plot(lr_sweep[lr]["loss_history"], linewidth=1.5, label=f"lr={lr}")
    ax.set_xlabel("Boosting Stage")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{name} - Loss Curves by Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_lr_loss.png")


def plot_confusion(y_true, y_pred, target_names, name):
    cm, _ = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(target_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(target_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_confusion.png")


def print_results(all_results):
    for name, res in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        print(f"Best n_estimators:    {res['best_n']}")
        print(f"Best max_depth:       {res['best_depth']}")
        print(f"Best learning_rate:   {res['best_lr']}")
        print(f"Test accuracy:        {accuracy(res['y_test'], res['y_pred']):.4f}")

        print("\nLearning-rate sweep:")
        for lr in sorted(res["lr_sweep"]):
            info = res["lr_sweep"][lr]
            print(f"  lr={lr:<5}  train={info['train_acc']:.4f}  test={info['test_acc']:.4f}")

        print("\nClassification report:")
        print(classification_report(res["y_test"], res["y_pred"],
                                    label_names=res["target_names"]))


if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        plot_n_estimators(res["n_vals"], res["n_train"], res["n_test"], name)
        plot_depth(res["d_vals"], res["d_train"], res["d_test"], name)
        plot_lr_accuracy(res["lr_sweep"], name)
        plot_lr_loss(res["lr_sweep"], name)
        plot_confusion(res["y_test"], res["y_pred"], res["target_names"], name)

    print_results(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
