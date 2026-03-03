import numpy as np


class SMO:
    """Simplified Sequential Minimal Optimization (SMO) solver for the
    dual SVM problem.

    Solves:
        max_α  Σ α_i − ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
        s.t.   0 ≤ α_i ≤ C   and   Σ α_i y_i = 0

    """

    def __init__(self, C=1.0, tol=1e-3, max_passes=100):
        """
        Parameters
        ----------
        C : float
            Regularization parameter (box constraint).
        tol : float
            Numerical tolerance for KKT violation checks.
        max_passes : int
            Maximum number of passes over the dataset without any
            alpha change before stopping.
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes


    def solve(self, K, y):
        """Run SMO on a precomputed kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n, n) – Gram matrix.
        y : ndarray of shape (n,)   – labels in {-1, +1}.

        Returns
        -------
        alphas : ndarray of shape (n,)  – Lagrange multipliers.
        b      : float                  – bias (intercept).
        """
        n = len(y)
        alphas = np.zeros(n)
        b = 0.0
        passes = 0

        while passes < self.max_passes:
            num_changed = 0

            for i in range(n):
                E_i = self._decision(K, y, alphas, b, i) - y[i]

                if not self._violates_kkt(alphas[i], y[i], E_i):
                    continue

                # pick j ≠ i randomly
                j = i
                while j == i:
                    j = np.random.randint(0, n)

                E_j = self._decision(K, y, alphas, b, j) - y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # compute bounds L, H
                L, H = self._bounds(alphas[i], alphas[j], y[i], y[j])
                if L >= H:
                    continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (E_i - E_j) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                # update bias
                b1 = (b - E_i
                       - y[i] * (alphas[i] - alpha_i_old) * K[i, i]
                       - y[j] * (alphas[j] - alpha_j_old) * K[i, j])
                b2 = (b - E_j
                       - y[i] * (alphas[i] - alpha_i_old) * K[i, j]
                       - y[j] * (alphas[j] - alpha_j_old) * K[j, j])

                if 0 < alphas[i] < self.C:
                    b = b1
                elif 0 < alphas[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        return alphas, b

    def _decision(self, K, y, alphas, b, idx):
        """Compute the SVM decision value for sample *idx*."""
        return float(np.sum(alphas * y * K[:, idx]) + b)

    def _violates_kkt(self, alpha_i, y_i, E_i):
        """Check whether sample i violates the KKT conditions."""
        r = y_i * E_i
        return ((r < -self.tol and alpha_i < self.C) or
                (r > self.tol and alpha_i > 0))

    def _bounds(self, alpha_i, alpha_j, y_i, y_j):
        """Compute the clipping bounds L, H for α_j."""
        if y_i != y_j:
            L = max(0.0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0.0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)
        return L, H


class SubGradientSVM:
    """Sub-gradient descent solver for the primal (linear) SVM.

    Minimises:  ½ ||w||² + C Σ max(0, 1 − y_i(w · x_i + b))

    This solver only supports the linear kernel.
    """

    def __init__(self, C=1.0, lr=0.001, max_iter=1000, tol=1e-6):
        """
        Parameters
        ----------
        C : float
            Regularization parameter.
        lr : float
            Learning rate.
        max_iter : int
            Maximum training iterations.
        tol : float
            Convergence tolerance on the loss.
        """
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, X, y):
        """Train a linear SVM via sub-gradient descent.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ndarray (n_samples,) – labels in {-1, +1}.

        Returns
        -------
        w : ndarray (n_features,) – weight vector.
        b : float                 – bias.
        loss_history : list       – hinge loss at each iteration.
        """
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        loss_history = []

        for t in range(1, self.max_iter + 1):
            margins = y * (X @ w + b)               # (n,)
            hinge_mask = margins < 1.0               # violated samples

            # hinge loss + regularization
            loss = 0.5 * np.dot(w, w) + self.C * np.sum(np.maximum(0, 1 - margins))
            loss_history.append(loss)

            # sub-gradients
            dw = w.copy()
            db = 0.0
            for i in range(n):
                if hinge_mask[i]:
                    dw -= self.C * y[i] * X[i]
                    db -= self.C * y[i]
            dw /= n
            db /= n

            # learning rate schedule: lr / sqrt(t)
            eta = self.lr / np.sqrt(t)
            w -= eta * dw
            b -= eta * db

            if t > 1 and abs(loss_history[-2] - loss_history[-1]) < self.tol:
                break

        return w, b, loss_history
