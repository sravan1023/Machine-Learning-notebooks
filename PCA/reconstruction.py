import numpy as np
import matplotlib.pyplot as plt


def reconstruct(X_scaled, pca, n_components):
    X_reduced = X_scaled @ pca.components_[:n_components].T
    X_reconstructed = X_reduced @ pca.components_[:n_components]
    return X_reconstructed


def get_sample_indices(y, n_samples=5):
    np.random.seed(42)
    indices = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        indices.append(np.random.choice(idx))
    return indices[:n_samples]


def plot_reconstruction(X_scaled, y, pca, scaler, image_shape,
                        target_names, cumulative_variance, sample_indices,
                        comp_values=(50, 150)):
    n_rows = len(sample_indices)
    n_cols = len(comp_values) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle('Original vs Reconstructed Faces', fontsize=16, fontweight='bold', y=1.02)

    for row, idx in enumerate(sample_indices):
        original = scaler.inverse_transform(X_scaled[idx].reshape(1, -1)).reshape(image_shape)
        axes[row, 0].imshow(original, cmap='gray')
        if row == 0:
            axes[row, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel(target_names[y[idx]].split()[-1])

        for col, n_comp in enumerate(comp_values, start=1):
            reconstructed_scaled = reconstruct(X_scaled[idx:idx+1], pca, n_comp)
            reconstructed = scaler.inverse_transform(reconstructed_scaled).reshape(image_shape)
            axes[row, col].imshow(reconstructed, cmap='gray')
            if row == 0:
                var_retained = cumulative_variance[n_comp - 1] * 100
                axes[row, col].set_title(f'{n_comp} PCs ({var_retained:.1f}%)',
                                         fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
