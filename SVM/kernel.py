import numpy as np


def linear(x1, x2):
    """Linear kernel:  K(x1, x2) = x1 · x2

    Parameters
    ----------
    x1, x2 : ndarray of shape (n_features,)

    Returns
    -------
    float
    """
    return float(np.dot(x1, x2))


def polynomial(x1, x2, degree=3, coef0=1.0):
    """Polynomial kernel:  K(x1, x2) = (x1 · x2 + coef0)^degree

    Parameters
    ----------
    x1, x2 : ndarray of shape (n_features,)
    degree  : int
    coef0   : float – independent term.

    Returns
    -------
    float
    """
    return float((np.dot(x1, x2) + coef0) ** degree)


def rbf(x1, x2, gamma=1.0):
    """Radial basis function (Gaussian) kernel:
    K(x1, x2) = exp(-gamma * ||x1 - x2||^2)

    Parameters
    ----------
    x1, x2 : ndarray of shape (n_features,)
    gamma   : float – kernel coefficient.

    Returns
    -------
    float
    """
    diff = x1 - x2
    return float(np.exp(-gamma * np.dot(diff, diff)))


def sigmoid_kernel(x1, x2, gamma=0.01, coef0=0.0):
    """Sigmoid (hyperbolic tangent) kernel:
    K(x1, x2) = tanh(gamma * x1 · x2 + coef0)

    Parameters
    ----------
    x1, x2 : ndarray of shape (n_features,)
    gamma   : float
    coef0   : float

    Returns
    -------
    float
    """
    return float(np.tanh(gamma * np.dot(x1, x2) + coef0))


# Kernel matrix utilities

def kernel_matrix(X, kernel_fn, **kwargs):
    """Compute the full kernel (Gram) matrix for dataset X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    kernel_fn : callable – one of the kernel functions above.
    **kwargs  : extra parameters forwarded to kernel_fn.

    Returns
    -------
    K : ndarray of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = kernel_fn(X[i], X[j], **kwargs)
            K[i, j] = val
            K[j, i] = val
    return K


def kernel_vector(X, x, kernel_fn, **kwargs):
    """Compute kernel between every row of X and a single vector x.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    x : ndarray of shape (n_features,)

    Returns
    -------
    k : ndarray of shape (n_samples,)
    """
    return np.array([kernel_fn(X[i], x, **kwargs) for i in range(X.shape[0])])


# Convenience lookup

def get_kernel(name, **kwargs):
    """Return a callable kernel_fn(x1, x2) with baked-in parameters.

    Parameters
    ----------
    name : str – 'linear', 'poly', 'rbf', or 'sigmoid'.

    Returns
    -------
    callable
    """
    if name == "linear":
        return linear
    elif name == "poly":
        degree = kwargs.get("degree", 3)
        coef0 = kwargs.get("coef0", 1.0)
        return lambda x1, x2: polynomial(x1, x2, degree=degree, coef0=coef0)
    elif name == "rbf":
        gamma = kwargs.get("gamma", 1.0)
        return lambda x1, x2: rbf(x1, x2, gamma=gamma)
    elif name == "sigmoid":
        gamma = kwargs.get("gamma", 0.01)
        coef0 = kwargs.get("coef0", 0.0)
        return lambda x1, x2: sigmoid_kernel(x1, x2, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(f"Unknown kernel: {name}")
