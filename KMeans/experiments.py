import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from kmeans import KMeansScratch
from metrics import inertia, silhouette_score


def load_datasets():
    X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, random_state=42)
    X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    datasets = {
        "Blobs": (StandardScaler().fit_transform(X_blobs), y_blobs),
        "Moons": (StandardScaler().fit_transform(X_moons), y_moons),
        "Iris":  (StandardScaler().fit_transform(X_iris), y_iris),
    }
    return datasets


def sweep_k(X, k_range=range(2, 11), init="kmeans++", n_init=10, random_state=42):
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeansScratch(n_clusters=k, init=init, n_init=n_init,
                           random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))

    return list(k_range), inertias, silhouettes


def compare_init_methods(X, k=4, n_init_values=(1, 10), random_state=42):
    results = {}
    for method in ["random", "kmeans++"]:
        for ni in n_init_values:
            km = KMeansScratch(n_clusters=k, init=method, n_init=ni,
                               random_state=random_state)
            km.fit(X)
            results[(method, ni)] = {
                "inertia": km.inertia_,
                "n_iter": km.n_iter_,
            }
    return results


def best_k_run(X, k, init="kmeans++", n_init=10, random_state=42):
    km = KMeansScratch(n_clusters=k, init=init, n_init=n_init,
                       random_state=random_state)
    km.fit(X)
    return km


def run_all():
    datasets = load_datasets()
    all_results = {}

    for name, (X, y_true) in datasets.items():
        k_vals, inertias, silhouettes = sweep_k(X)
        best_k = k_vals[np.argmax(silhouettes)]

        km_best = best_k_run(X, best_k)
        init_compare = compare_init_methods(X, k=best_k)

        all_results[name] = {
            "X": X,
            "y_true": y_true,
            "k_vals": k_vals,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "best_k": best_k,
            "km_best": km_best,
            "init_compare": init_compare,
        }

    return all_results
