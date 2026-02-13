"""
PCA-based Dimensionality Reduction and Reconstruction Pipeline
Using the LFW (Labeled Faces in the Wild) Face Dataset
"""

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler

# Data Preparation

# Load the LFW dataset (min 70 images per person for a manageable subset)
print("Loading LFW dataset (this may take a moment on first run)...")
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw.data            # feature matrix (flattened pixel values)
y = lfw.target           # labels
target_names = lfw.target_names
image_shape = lfw.images[0].shape  # (height, width) of each face image

n_samples, n_features = X.shape
print(f"  Dataset Summary")
print()
print(f"  Number of samples  : {n_samples}")
print(f"  Number of features : {n_features} pixels ({image_shape[0]}h x {image_shape[1]}w)")
print(f"  Number of classes  : {len(target_names)}")
print(f"  Class names        : {', '.join(target_names)}")

# Normalize features (mean centering + scaling)
# PCA requires at least mean-centering; StandardScaler also unit-variance scales
# so each pixel contributes equally regardless of its brightness range.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Normalization applied (mean centering + unit-variance scaling).")
print(f"  Per-feature mean after scaling : {X_scaled.mean(axis=0).mean():.2e}  (≈ 0)")
print(f"  Per-feature std  after scaling : {X_scaled.std(axis=0).mean():.4f}  (≈ 1)")
