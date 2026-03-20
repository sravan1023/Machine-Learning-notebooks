import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import run_all
from metrics import confusion_matrix, classification_report

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def _safe_name(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _save(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_param_sweep(sweep, best_param, name):
    params = list(sweep.keys())
    train_accs = [sweep[p]["train_acc"] for p in params]
    test_accs = [sweep[p]["test_acc"] for p in params]

    x = np.arange(len(params))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, train_accs, width, label="Train")
    ax.bar(x + width / 2, test_accs, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in params])
    ax.set_xlabel("Hyperparameter")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    ax.set_title(f"{name} - Hyperparameter Sweep (best={best_param})")
    plt.tight_layout()
    _save(fig, f"{_safe_name(name)}_sweep.png")


def plot_conf_matrix(y_true, y_pred, target_names, name):
    cm, _ = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(target_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(target_names)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    _save(fig, f"{_safe_name(name)}_confusion.png")


def print_results(all_results):
    for name, res in all_results.items():
        print(f"\n{'=' * 70}")
        print(name)
        print(f"{'=' * 70}")
        print(f"Algorithm:      {res['algorithm']}")
        print(f"Best parameter: {res['best_param']}")
        print(f"Test accuracy:  {res['test_acc']:.4f}")
        print("\nSweep:")
        for p in sorted(res["sweep"].keys()):
            info = res["sweep"][p]
            print(f"  {p:<10} train={info['train_acc']:.4f}  test={info['test_acc']:.4f}")
        print("\nClassification report:")
        print(classification_report(res["y_test"], res["y_pred"], label_names=res["target_names"]))


if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        plot_param_sweep(res["sweep"], res["best_param"], name)
        plot_conf_matrix(res["y_test"], res["y_pred"], res["target_names"], name)

    print_results(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
