import numpy as np
import matplotlib.pyplot as plt
from reconstruction import reconstruct


def add_noise(X_scaled, noise_level=1.5):
    np.random.seed(42)
    X_noisy = X_scaled + noise_level * np.random.randn(*X_scaled.shape)
    return X_noisy


def denoise(X_noisy, pca, n_components=150):
    return reconstruct(X_noisy, pca, n_components)


def plot_noise_reduction(X_scaled, X_noisy, X_denoised, scaler, image_shape,
                         sample_indices, noise_level, n_components):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle('PCA Noise Reduction', fontsize=16, fontweight='bold')

    row_labels = ['Original', f'Noisy (σ={noise_level})', f'Denoised ({n_components} PCs)']

    for col, idx in enumerate(sample_indices):
        original = scaler.inverse_transform(X_scaled[idx].reshape(1, -1)).reshape(image_shape)
        axes[0, col].imshow(original, cmap='gray')
        if col == 0:
            axes[0, col].set_ylabel(row_labels[0], fontsize=12, fontweight='bold')

        noisy_img = scaler.inverse_transform(X_noisy[idx].reshape(1, -1)).reshape(image_shape)
        axes[1, col].imshow(noisy_img, cmap='gray')
        if col == 0:
            axes[1, col].set_ylabel(row_labels[1], fontsize=12, fontweight='bold')

        denoised_img = scaler.inverse_transform(X_denoised[idx].reshape(1, -1)).reshape(image_shape)
        axes[2, col].imshow(denoised_img, cmap='gray')
        if col == 0:
            axes[2, col].set_ylabel(row_labels[2], fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('noise_reduction.png', dpi=150, bbox_inches='tight')
    plt.show()
