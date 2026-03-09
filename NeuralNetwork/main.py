import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import run_all, accuracy, confusion_matrix, classification_report

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# Plot helpers
def _save(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


# XOR plots
def plot_xor_loss(loss_history):
    """Training loss curve for the XOR experiment."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, linewidth=2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("XOR – Training Loss Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "xor_loss_curve.png")


def plot_xor_decision_boundary(model, X, y):
    """2-D decision boundary for XOR."""
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdYlBu", alpha=0.8)
    fig.colorbar(contour, ax=ax, label="P(class 1)")

    colors = ["#d62728", "#2ca02c"]
    for cls in [0, 1]:
        mask = y.ravel() == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls],
                   edgecolors="k", s=200, zorder=5,
                   label=f"Class {cls}")

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("XOR – Decision Boundary")
    ax.legend()
    plt.tight_layout()
    _save(fig, "xor_decision_boundary.png")


# Digits plots
def plot_digits_loss(loss_history):
    """Training loss curve for the Digits experiment."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, linewidth=2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical Cross-Entropy Loss")
    ax.set_title("Digits – Training Loss Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "digits_loss_curve.png")


def plot_confusion_matrix(y_true, y_pred, target_names, name):
    """Confusion matrix heatmap."""
    cm, labels = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
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
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} – Confusion Matrix")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_confusion.png")


def plot_lr_comparison(lr_sweep, name):
    """Loss curves for different learning rates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for lr in sorted(lr_sweep.keys()):
        ax.plot(lr_sweep[lr]["loss_history"], linewidth=1.5,
                label=f"lr={lr}")
    ax.set_xlabel("Epoch")
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
    ax.bar(x - width / 2, train_accs, width, label="Train", color="steelblue")
    ax.bar(x + width / 2, test_accs, width, label="Test", color="coral")
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


def plot_hidden_size_comparison(hidden_sweep, name):
    """Loss curves for different hidden layer sizes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in sorted(hidden_sweep.keys()):
        ax.plot(hidden_sweep[h]["loss_history"], linewidth=1.5,
                label=f"hidden={h}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} – Loss vs Hidden Layer Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_hidden_comparison.png")


def plot_hidden_size_accuracy(hidden_sweep, name):
    """Bar chart of test accuracy per hidden layer size."""
    sizes = sorted(hidden_sweep.keys())
    test_accs = [hidden_sweep[h]["test_acc"] for h in sizes]
    train_accs = [hidden_sweep[h]["train_acc"] for h in sizes]

    x = np.arange(len(sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, train_accs, width, label="Train", color="steelblue")
    ax.bar(x + width / 2, test_accs, width, label="Test", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Hidden Layer Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs Hidden Layer Size")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{name.lower().replace(' ', '_')}_hidden_accuracy.png")


def plot_sample_digits(X_test, y_test, y_pred, n=16):
    """Grid of sample digit images with predictions."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i >= n or i >= len(y_test):
            ax.axis("off")
            continue
        img = X_test[i].reshape(8, 8)
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        colour = "green" if y_pred[i] == y_test[i] else "red"
        ax.set_title(f"pred={y_pred[i]}  true={y_test[i]}",
                     fontsize=9, color=colour)
        ax.axis("off")
    plt.suptitle("Digits – Sample Predictions", fontsize=13)
    plt.tight_layout()
    _save(fig, "digits_sample_predictions.png")


# Console summary
def print_summary(results):
    xor = results["xor"]
    digits = results["digits"]

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    print(f"\nXOR")
    print(f"  Final loss:   {xor['loss_history'][-1]:.6f}")
    print(f"  Accuracy:     {accuracy(xor['y'].ravel(), xor['y_pred']):.4f}")

    print(f"\nDigits (8×8)")
    print(f"  Train acc:    {digits['train_acc']:.4f}")
    print(f"  Test acc:     {digits['test_acc']:.4f}")
    print(f"  Final loss:   {digits['loss_history'][-1]:.6f}")

    print(f"\nLearning-rate sweep (Digits):")
    for lr in sorted(results["lr_sweep"]):
        info = results["lr_sweep"][lr]
        print(f"  lr={lr:<6}  train={info['train_acc']:.4f}  "
              f"test={info['test_acc']:.4f}")

    print(f"\nHidden-size sweep (Digits):")
    for h in sorted(results["hidden_sweep"]):
        info = results["hidden_sweep"][h]
        print(f"  hidden={h:<4}  train={info['train_acc']:.4f}  "
              f"test={info['test_acc']:.4f}")


# Entry point 
if __name__ == "__main__":
    results = run_all()

    # XOR plots
    plot_xor_loss(results["xor"]["loss_history"])
    plot_xor_decision_boundary(results["xor"]["model"],
                               results["xor"]["X"],
                               results["xor"]["y"])

    # Digits plots
    plot_digits_loss(results["digits"]["loss_history"])
    plot_confusion_matrix(results["digits"]["y_test"],
                          results["digits"]["y_pred"],
                          results["digits"]["target_names"],
                          "Digits")
    plot_sample_digits(results["digits"]["X_test"],
                       results["digits"]["y_test"],
                       results["digits"]["y_pred"])

    # Sweep plots
    plot_lr_comparison(results["lr_sweep"], "Digits")
    plot_lr_accuracy(results["lr_sweep"], "Digits")
    plot_hidden_size_comparison(results["hidden_sweep"], "Digits")
    plot_hidden_size_accuracy(results["hidden_sweep"], "Digits")

    print_summary(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
