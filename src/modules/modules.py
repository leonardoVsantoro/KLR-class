import pandas as pd # type: ignore
import numpy as np # type: ignore
import random # type: ignore
from scipy.stats import wishart # type: ignore
from scipy.spatial.distance import cdist # type: ignore
from numpy.linalg import LinAlgError # type: ignore
import os # type: ignore
from datetime import datetime # type: ignore
from scipy import linalg as LA# type: ignore
from joblib import Parallel, delayed # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from scipy.spatial import distance_matrix # type: ignore
from scipy.sparse.csgraph import minimum_spanning_tree# type: ignore
from scipy.spatial import distance# type: ignore
from scipy.linalg import det
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.ticker import ScalarFormatter # type: ignore
from tqdm import tqdm# type: ignore
from scipy.ndimage import gaussian_filter# type: ignore
import math
from scipy.stats import vonmises_fisher

from matplotlib.colors import ListedColormap
from matplotlib import colormaps as mpl_colormaps
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from prettytable import PrettyTable
import time
from traitlets import Bunch

