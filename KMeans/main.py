import numpy as np
import matplotlib.pyplot as plt
from experiments import run_all


def plot_clusters(X, km, name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', s=30, alpha=0.7)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=1.5)
    ax.set_title(f'{name} - K={km.n_clusters}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.tight_layout()
    plt.savefig(f'plots/{name.lower()}_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_elbow(k_vals, inertias, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, inertias, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title(f'{name} - Elbow Plot')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{name.lower()}_elbow.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_silhouette(k_vals, silhouettes, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, silhouettes, 'go-', linewidth=2, markersize=6)
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title(f'{name} - Silhouette Score vs k')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{name.lower()}_silhouette.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_convergence(km, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(km.inertia_history_) + 1), km.inertia_history_,
            'ro-', linewidth=2, markersize=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Inertia')
    ax.set_title(f'{name} - Convergence (K={km.n_clusters})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{name.lower()}_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_results(all_results):
    for name, res in all_results.items():
        km = res["km_best"]
        best_k = res["best_k"]
        best_sil = max(res["silhouettes"])

        print(f"\n--- {name} ---")
        print(f"Best k (by silhouette): {best_k}")
        print(f"Silhouette score:       {best_sil:.4f}")
        print(f"Best inertia:           {km.inertia_:.4f}")
        print(f"Iterations:             {km.n_iter_}")
        print(f"Centroids:\n{np.round(km.cluster_centers_, 4)}")

        print(f"\nInit method comparison (k={best_k}):")
        for (method, ni), vals in res["init_compare"].items():
            print(f"  {method:10s} n_init={ni:2d}  "
                  f"inertia={vals['inertia']:.4f}  iters={vals['n_iter']}")


if __name__ == "__main__":
    results = run_all()

    for name, res in results.items():
        X = res["X"]
        km = res["km_best"]

        plot_clusters(X, km, name)
        plot_elbow(res["k_vals"], res["inertias"], name)
        plot_silhouette(res["k_vals"], res["silhouettes"], name)
        plot_convergence(km, name)

    print_results(results)
