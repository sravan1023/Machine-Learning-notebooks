from data_preparation import load_data, normalize
from pca_reduction import apply_pca, get_components_for_variance, plot_scree
from eigenfaces import plot_eigenfaces
from reconstruction import get_sample_indices, plot_reconstruction
from noise_reduction import add_noise, denoise, plot_noise_reduction
from classification import run_classification, plot_classification_results

# 1. Data Preparation
X, y, target_names, image_shape, n_samples, n_features = load_data()
X_scaled, scaler = normalize(X)

# 2. PCA Dimensionality Reduction
pca, X_pca, explained_variance_ratio, cumulative_variance = apply_pca(X_scaled, n_samples, n_features)

for t in [0.80, 0.90, 0.95]:
    n_comp, actual = get_components_for_variance(cumulative_variance, t)
    print(f"{t*100:.0f}% variance: {n_comp} components (actual: {actual*100:.2f}%)")

plot_scree(explained_variance_ratio, cumulative_variance)

# 3. Eigenfaces Visualization
plot_eigenfaces(pca, image_shape, explained_variance_ratio)

# 4. Reconstruction
sample_indices = get_sample_indices(y)
plot_reconstruction(X_scaled, y, pca, scaler, image_shape,
                    target_names, cumulative_variance, sample_indices)

# 5. Noise Reduction
noise_level = 1.5
n_denoise_components = 150
X_noisy = add_noise(X_scaled, noise_level)
X_denoised = denoise(X_noisy, pca, n_denoise_components)
plot_noise_reduction(X_scaled, X_noisy, X_denoised, scaler, image_shape,
                     sample_indices, noise_level, n_denoise_components)

# 6. Classification
results = run_classification(X_scaled, y, target_names, n_features)
print(results['report'])
plot_classification_results(results)
