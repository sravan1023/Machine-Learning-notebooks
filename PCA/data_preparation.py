import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler


def load_data():
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    X = lfw.data
    y = lfw.target
    target_names = lfw.target_names
    image_shape = lfw.images[0].shape
    n_samples, n_features = X.shape

    return X, y, target_names, image_shape, n_samples, n_features


def normalize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
