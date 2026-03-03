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


# Kernel comparison bar chart
def plot_kernel_comparison(kernel_cmp, name):
    kernels = list(kernel_cmp.keys())
    test_accs = [kernel_cmp[k]["test_acc"] for k in kernels]
    macro_f1s = [kernel_cmp[k]["macro_f1"] for k in kernels]

    x = np.arange(len(kernels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, test_accs, width, label="Test Accuracy", color="steelblue")
    ax.bar(x + width / 2, macro_f1s, width, label="Macro F1", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.set_ylabel("Score")
    ax.set_title(f"{name} – Kernel Comparison")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_kernel_cmp.png")


# C sweep
def plot_c_sweep(c_sweep, name):
    cs = sorted(c_sweep.keys())
    train_accs = [c_sweep[c]["train_acc"] for c in cs]
    test_accs = [c_sweep[c]["test_acc"] for c in cs]
    n_svs = [c_sweep[c]["n_support"] for c in cs]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.semilogx(cs, train_accs, "bo-", lw=2, ms=6, label="Train Acc")
    ax1.semilogx(cs, test_accs, "ro-", lw=2, ms=6, label="Test Acc")
    ax1.set_xlabel("C")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.semilogx(cs, n_svs, "g^--", lw=1.5, ms=6, label="#SVs")
    ax2.set_ylabel("Number of Support Vectors", color="green")
    ax2.legend(loc="lower right")

    ax1.set_title(f"{name} – Accuracy & #SVs vs C")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_c_sweep.png")


# Gamma sweep
def plot_gamma_sweep(gamma_sweep, name):
    gs = sorted(gamma_sweep.keys())
    train_accs = [gamma_sweep[g]["train_acc"] for g in gs]
    test_accs = [gamma_sweep[g]["test_acc"] for g in gs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(gs, train_accs, "bo-", lw=2, ms=6, label="Train")
    ax.semilogx(gs, test_accs, "ro-", lw=2, ms=6, label="Test")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs Gamma (RBF kernel)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_gamma_sweep.png")


# Polynomial degree sweep
def plot_degree_sweep(degree_sweep, name):
    ds = sorted(degree_sweep.keys())
    train_accs = [degree_sweep[d]["train_acc"] for d in ds]
    test_accs = [degree_sweep[d]["test_acc"] for d in ds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ds, train_accs, "bo-", lw=2, ms=6, label="Train")
    ax.plot(ds, test_accs, "ro-", lw=2, ms=6, label="Test")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Accuracy vs Polynomial Degree")
    ax.set_xticks(ds)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_degree_sweep.png")


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
    ax.set_title(f"{name} – Confusion Matrix (SVM)")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_confusion.png")


# CV accuracy bars
def plot_cv_accuracy(cv_accs, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = np.arange(1, len(cv_accs) + 1)
    ax.bar(folds, cv_accs, color="steelblue", edgecolor="black")
    ax.axhline(cv_accs.mean(), color="red", ls="--", label=f"Mean = {cv_accs.mean():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{name} – Cross-Validation Accuracy (SVM)")
    ax.set_xticks(folds)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_cv_accuracy.png")


# Decision boundary (first 2 features)
def plot_decision_boundary(X_train, y_train, model, name):
    from svm import SVC

    X2 = X_train[:, :2]
    svm2 = SVC(C=model.C, kernel=model.kernel, degree=model.degree,
               gamma=model.gamma, coef0=model.coef0, random_state=42)
    svm2.fit(X2, y_train)

    h = 0.1
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svm2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X2[:, 0], X2[:, 1], c=y_train, cmap="viridis",
               edgecolors="k", s=30)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{name} – Decision Boundary (SVM, first 2 features)")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_boundary.png")


# Number of support vectors per kernel
def plot_n_support_vectors(kernel_cmp, name):
    kernels = list(kernel_cmp.keys())
    n_svs = [kernel_cmp[k]["n_support"] for k in kernels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(kernels, n_svs, color="mediumpurple", edgecolor="black")
    ax.set_ylabel("Number of Support Vectors")
    ax.set_title(f"{name} – Support Vectors per Kernel")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, f"{_slug(name)}_n_support.png")


# Console report
def print_results(all_results):
    for name, res in all_results.items():
        print()
        print(f"  {name}")
        print()
        print(f"Best kernel:       {res['best_kernel']}")
        print(f"Best C:            {res['best_C']}")
        print(f"Best gamma:        {res['best_gamma']}")
        print(f"Test accuracy:     {accuracy(res['y_test'], res['y_pred']):.4f}")
        print(f"CV accuracy:       {res['cv_accs'].mean():.4f} "
              f"(± {res['cv_accs'].std():.4f})")
        print(f"Support vectors:   {res['best_model'].get_n_support()}")

        print(f"\nKernel comparison:")
        for k, v in res["kernel_cmp"].items():
            print(f"  {k:>10s}  acc={v['test_acc']:.4f}  "
                  f"f1={v['macro_f1']:.4f}  #sv={v['n_support']}")

        print(f"\nC sweep ({res['best_kernel']} kernel):")
        for c in sorted(res["c_sweep"]):
            v = res["c_sweep"][c]
            print(f"  C={c:<8}  train={v['train_acc']:.4f}  "
                  f"test={v['test_acc']:.4f}  #sv={v['n_support']}")

        print(f"\nGamma sweep (RBF, C={res['best_C']}):")
        for g in sorted(res["gamma_sweep"]):
            v = res["gamma_sweep"][g]
            print(f"  γ={g:<8}  train={v['train_acc']:.4f}  "
                  f"test={v['test_acc']:.4f}  #sv={v['n_support']}")

        print(f"\nPolynomial degree sweep:")
        for d in sorted(res["degree_sweep"]):
            v = res["degree_sweep"][d]
            print(f"  deg={d}  train={v['train_acc']:.4f}  "
                  f"test={v['test_acc']:.4f}  #sv={v['n_support']}")

        print(f"\nClassification report:")
        print(classification_report(res["y_test"], res["y_pred"],
                                    label_names=res["target_names"]))


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        plot_kernel_comparison(res["kernel_cmp"], name)
        plot_c_sweep(res["c_sweep"], name)
        plot_gamma_sweep(res["gamma_sweep"], name)
        plot_degree_sweep(res["degree_sweep"], name)
        plot_confusion_matrix(res["y_test"], res["y_pred"],
                              res["target_names"], name)
        plot_cv_accuracy(res["cv_accs"], name)
        plot_decision_boundary(res["X_train"], res["y_train"],
                               res["best_model"], name)
        plot_n_support_vectors(res["kernel_cmp"], name)

    print_results(results)
    print(f"\nPlots saved to {PLOT_DIR}/")
