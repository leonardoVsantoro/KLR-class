
import os
import sys
script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from src.modules import *
from src.classification import *
# ------ 0/1 CLASS
def make_linearly_separable(n_samples= 100, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, noise = 0):
    """
    Generate a linearly separable dataset.
    This function generates a synthetic dataset using sklearn's make_classification function
    Parameters:
    n_samples (int): Total number of samples to generate (default is 100).
    n_features (int): Total number of features (default is 2).
    n_redundant (int): Number of redundant features (default is 0).
    n_informative (int): Number of informative features (default is 2).
    random_state (int): Random seed for reproducibility (default is 1).
    n_clusters_per_class (int): Number of clusters per class (default is 1).
    noise (float): Standard deviation of Gaussian noise added to the data (default is 0).
    Returns:
    tuple: A tuple containing the generated features (X) and labels (y).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=n_redundant,
        n_informative=n_informative,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state
    )
    rng = np.random.RandomState(random_state)
    X += noise * rng.standard_normal(size=X.shape)
    return (X, y)

def make_xor(n_samples = 100, random_state=1):
    """
    Generate XOR dataset.
    This function generates a dataset with two features and labels based on the XOR logic.
    Parameters:
    n_samples (int): Total number of samples to generate (default is 100).
    random_state (int): Random seed for reproducibility (default is 1).
    Returns:
    tuple: A tuple containing the generated features (X_xor) and labels (y_xor).
    """
    rng = np.random.RandomState(random_state)
    X_xor = rng.randn(n_samples, 2)
    X_xor += 2 * rng.standard_normal(size=X_xor.shape)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0).astype(int)  
    return (X_xor, y_xor)  

# ----- MULTICLASS
def make4moons(n_samples=100, noise=0.15, shift = (2.5, 1.5), random_state=1):
    """
    Generate 4-class moons dataset.
    This function generates a dataset with three classes using the make_moons function from sklearn,
    shifting the second class to create a distinct separation.
    Parameters:
    n_samples (int): Total number of samples to generate (default is 100).
    noise (float): Standard deviation of Gaussian noise added to the data (default is 0.15).
    shift (tuple): Shift applied to the second class to separate it from the first (default is (2.5, 1.5)).
    random_state (int): Random seed for reproducibility (default is 1).
    Returns:
    tuple: A tuple containing the generated features (X_moons) and labels (y_moons).
    """
    rng = np.random.RandomState(random_state)   
    X1, y1 = make_moons(n_samples=100, noise=noise, random_state=random_state)
    X2, y2 = make_moons(n_samples=100, noise=noise, random_state=random_state)
    X2 +=  [shift[0], shift[1]]
    X_moons = np.vstack((X1, X2))
    y_moons = np.hstack((y1, y2 + 2))  # classes: 0,1,2
    return (X_moons, y_moons)


def make4circles(n_samples=100, noise = 0.1, factor = .5, shift = (3,3), random_state=1):
    """
    Generate 4-class circles dataset.
    This function generates a dataset with four classes using the make_circles function from sklearn,
    shifting the second class to create a distinct separation.
    Parameters:
    n_samples (int): Total number of samples to generate for each class (default is 100).
    noise (float): Standard deviation of Gaussian noise added to the data (default is 0.1).
    factor (float): Scaling factor for the inner circle (default is 0.5).
    shift (tuple): Shift applied to the second class to separate it from the first (default is (3, 3)).
    random_state (int): Random seed for reproducibility (default is 1).
    Returns:
    tuple: A tuple containing the generated features (X_circles) and labels (y_circles).
    """
    rng = np.random.RandomState(random_state)
    X1, y1 = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    X2, y2 = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    X2 += [shift[0], shift[1]]
    X_circles = np.vstack((X1, X2))
    y_circles = np.hstack((y1, y2 + 2))  # classes: 0,1,2,3
    return (X_circles, y_circles)

def make4XOR(n_samples = 100, random_state=1):
    """
    Generate 4-class XOR dataset.
    This function generates a dataset with four classes based on XOR logic,
    where each class is defined by the combination of two binary features.
    Parameters:
    n_samples (int): Total number of samples to generate (default is 100).
    random_state (int): Random seed for reproducibility (default is 1).
    Returns:
    tuple: A tuple containing the generated features (X_xor) and labels (y_xor4).
    """
    rng = np.random.RandomState(random_state)
    X_xor = rng.randn(n_samples, 2)
    X_xor += 2 * rng.standard_normal(size=X_xor.shape)
    y_xor4 = 2 * (X_xor[:, 0] > 0).astype(int) + (X_xor[:, 1] > 0).astype(int)  # 0–3
    return (X_xor, y_xor4)

