import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles


# ------ 2-class datasets -----------------------------------------------

def make_linearly_separable(
    n_samples=100, n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=1, noise=0,
):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_redundant=n_redundant, n_informative=n_informative,
        n_clusters_per_class=n_clusters_per_class, random_state=random_state,
    )
    if noise > 0:
        rng = np.random.RandomState(random_state)
        X += noise * rng.standard_normal(size=X.shape)
    return X, y


def make_xor(n_samples=100, random_state=1):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2) + 2 * rng.standard_normal((n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    return X, y


# ------ multiclass datasets --------------------------------------------

def make4moons(n_samples=100, noise=0.15, shift=(2.5, 1.5), random_state=1):
    X1, y1 = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X2, y2 = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X2 += np.array(shift)
    return np.vstack((X1, X2)), np.hstack((y1, y2 + 2))


def make4circles(n_samples=100, noise=0.1, factor=0.5, shift=(3, 3), random_state=1):
    X1, y1 = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    X2, y2 = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    X2 += np.array(shift)
    return np.vstack((X1, X2)), np.hstack((y1, y2 + 2))


def make4XOR(n_samples=100, random_state=1):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2) + 2 * rng.standard_normal((n_samples, 2))
    y = 2 * (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    return X, y
