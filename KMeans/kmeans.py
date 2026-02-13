import numpy as np


class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 random_state=None, init="kmeans++", n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.inertia_history_ = None

    def _init_centroids(self, X, rng):
        n_samples = X.shape[0]

        if self.init == "random":
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()

        elif self.init == "kmeans++":
            centroids = np.empty((self.n_clusters, X.shape[1]))
            centroids[0] = X[rng.randint(n_samples)]

            for k in range(1, self.n_clusters):
                dists = np.min(
                    np.array([np.sum((X - c) ** 2, axis=1) for c in centroids[:k]]),
                    axis=0
                )
                probs = dists / dists.sum()
                centroids[k] = X[rng.choice(n_samples, p=probs)]

            return centroids

        else:
            raise ValueError(f"Unknown init method: {self.init}")

    @staticmethod
    def _assign_clusters(X, centroids):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
        return np.argmin(dists, axis=0)

    @staticmethod
    def _compute_inertia(X, labels, centroids):
        return sum(np.sum((X[labels == k] - centroids[k]) ** 2)
                   for k in range(len(centroids)))

    def _single_run(self, X, rng):
        centroids = self._init_centroids(X, rng)
        inertia_history = []

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_clusters(X, centroids)

            # handle empty clusters
            for k in range(self.n_clusters):
                if np.sum(labels == k) == 0:
                    centroids[k] = X[rng.randint(X.shape[0])]
                    labels = self._assign_clusters(X, centroids)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
                for k in range(self.n_clusters)
            ])

            inertia = self._compute_inertia(X, labels, new_centroids)
            inertia_history.append(inertia)

            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids

            if shift < self.tol:
                break

        return centroids, labels, inertia, iteration, inertia_history

    def fit(self, X):
        best_inertia = np.inf
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter, history = self._single_run(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centroids
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_iter_ = n_iter
                self.inertia_history_ = history

        return self

    def predict(self, X):
        return self._assign_clusters(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
