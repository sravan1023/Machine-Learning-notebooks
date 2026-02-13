import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def apply_pca(X_scaled, n_samples, n_features):
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return pca, X_pca, explained_variance_ratio, cumulative_variance


def get_components_for_variance(cumulative_variance, threshold):
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    actual = cumulative_variance[n_comp - 1]
    return n_comp, actual


def plot_scree(explained_variance_ratio, cumulative_variance):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, len(explained_variance_ratio) + 1),
                 explained_variance_ratio, 'bo-', linewidth=2, markersize=4)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, min(50, len(explained_variance_ratio)))

    axes[1].plot(range(1, len(cumulative_variance) + 1),
                 cumulative_variance, 'ro-', linewidth=2, markersize=4)
    axes[1].axhline(y=0.80, color='g', linestyle='--', label='80%')
    axes[1].axhline(y=0.90, color='b', linestyle='--', label='90%')
    axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, min(200, len(cumulative_variance)))

    plt.tight_layout()
    plt.savefig('scree_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
