"""
bench_tabular.py
================
Benchmarks KLRClassifier and BaggingKLRClassifier against standard sklearn
classifiers on Iris, Breast Cancer, and Digits, plus:

  - Regularisation path  (ridge sweep, absolute and spectral)
  - Bandwidth sensitivity sweep
  - Bagging spectral-pruning table (bag_size × n_estimators)

Memory safety
-------------
All BaggingKLRClassifier runs use n_jobs=1 (serial bags) and explicit
gc.collect() between experiments.  store_covariance=False (default) halves
persistent memory per estimator from 2·C·n²·8 to C·n²·8 bytes.

Usage
-----
    cd <project-root>
    python3 tests/bench_tabular.py | tee tests/out/bench_tabular.txt
"""

import sys, os, gc, time, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import numpy as np
import psutil
from prettytable import PrettyTable
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from src.classification import KLRClassifier, BaggingKLRClassifier


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def mem_gb():
    return psutil.virtual_memory().available / 1e9

def require_gb(needed: float, label: str = '') -> bool:
    """Return True and print a warning if headroom < needed; caller should skip."""
    free = mem_gb()
    if free < needed:
        tag = f' [{label}]' if label else ''
        print(f'  SKIP{tag}: only {free:.1f} GB free, need {needed:.1f} GB')
        return True   # means "should skip"
    return False


def section(title: str):
    print(f'\n{"="*65}')
    print(title)
    print('='*65)


# ---------------------------------------------------------------------------
# 1. Classifier comparison
# ---------------------------------------------------------------------------

def compare_classifiers(name, X, y, classifiers, test_size=0.2, rs=42):
    y = LabelEncoder().fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rs)
    X_val, X_te2, y_val, y_te2 = train_test_split(X_te, y_te, test_size=0.5, random_state=rs)

    section(f'{name}  |  {X.shape[0]}×{X.shape[1]}  |  '
            f'{len(np.unique(y))} classes  |  '
            f'train={len(X_tr)} val={len(X_val)} test={len(X_te2)}')

    t = PrettyTable(['Classifier', 'Train', 'Val', 'Test', 'Time (s)'])
    for cname, clf in classifiers.items():
        t0 = time.time()
        if hasattr(clf, 'tune'):
            clf = clf.tune(X_tr, y_tr, X_val, y_val, verbose=False)
        else:
            clf = clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        t.add_row([cname,
                   f'{clf.score(X_tr,  y_tr):.3f}',
                   f'{clf.score(X_val, y_val):.3f}',
                   f'{clf.score(X_te2, y_te2):.3f}',
                   f'{elapsed:.2f}'])
        del clf; gc.collect()
    print(t)


classifiers = {
    'KLR (tuned)':   KLRClassifier(),
    'KLR (default)': KLRClassifier(ridge=1e-3, band_factor=1.0),
    'KNN':           KNeighborsClassifier(5),
    'SVM (RBF)':     SVC(probability=True),
    'QDA':           QuadraticDiscriminantAnalysis(),
    'Random Forest': RandomForestClassifier(100, random_state=0),
    'MLP':           MLPClassifier(max_iter=500, random_state=0),
    'Naive Bayes':   GaussianNB(),
}

compare_classifiers('IRIS',          *load_iris(return_X_y=True),          classifiers)
compare_classifiers('BREAST CANCER', *load_breast_cancer(return_X_y=True), classifiers)
compare_classifiers('DIGITS',        *load_digits(return_X_y=True),        classifiers)

# ---------------------------------------------------------------------------
# 2. Regularisation path  (Digits)
# ---------------------------------------------------------------------------

X, y = load_digits(return_X_y=True)
y = LabelEncoder().fit_transform(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
X_val, X_te2, y_val, y_te2 = train_test_split(X_te, y_te, test_size=0.5, random_state=0)

section('REGULARISATION PATH  |  Digits  |  fixed band_factor=1.0')
print('  Small ridge -> sharp boundary (approaches perfect classifier);')
print('  Large ridge -> kernel nearest-centroid (ignores covariance).\n')

for mode in ('absolute', 'spectral'):
    print(f'ridge_mode = {mode!r}:')
    t = PrettyTable(['ridge', 'Train', 'Val', 'Test', 'effective_ridge'])
    for r in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        clf = KLRClassifier(ridge=r, band_factor=1.0, ridge_mode=mode).fit(X_tr, y_tr)
        t.add_row([f'{r:.0e}',
                   f'{clf.score(X_tr,  y_tr):.3f}',
                   f'{clf.score(X_val, y_val):.3f}',
                   f'{clf.score(X_te2, y_te2):.3f}',
                   f'{clf.effective_ridge_:.4e}'])
        del clf; gc.collect()
    print(t)

# ---------------------------------------------------------------------------
# 3. Bandwidth sensitivity  (Digits)
# ---------------------------------------------------------------------------

section('BANDWIDTH SENSITIVITY  |  Digits  |  fixed ridge=1e-3')
t = PrettyTable(['band_factor', 'Train', 'Val', 'Test'])
for bf in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    clf = KLRClassifier(ridge=1e-3, band_factor=bf).fit(X_tr, y_tr)
    t.add_row([f'{bf}',
               f'{clf.score(X_tr,  y_tr):.3f}',
               f'{clf.score(X_val, y_val):.3f}',
               f'{clf.score(X_te2, y_te2):.3f}'])
    del clf; gc.collect()
print(t)

# ---------------------------------------------------------------------------
# 4. Bagging spectral-pruning study  (Digits)
# ---------------------------------------------------------------------------
# Memory budget per run:
#   stored  = B * C * m^2 * 8  bytes  (chols only, store_covariance=False)
#   peak    = (C+1) * m^2 * 8  bytes  (kernel + C covs during one bag fit)
# With n_jobs=1 only ONE bag is fitted at a time -> peak is tiny.

section('BAGGING SPECTRAL PRUNING  |  Digits  |  n_jobs=1')
print('  Smaller bag_size -> lower covariance rank -> stronger pruning.')
print('  n_jobs=1: bags fitted sequentially; peak mem per step is O(m^2).')
print('  store_covariance=False: Cholesky only; halves persistent mem.\n')

n_tr = len(X_tr)
C = len(np.unique(y_tr))
t2 = PrettyTable(['bag_size', 'n_est', 'm/n_tr', 'stored_GB', 'Test', 'Time'])

# cap bag_size at 0.5 to keep stored memory bounded
for bs in [0.05, 0.10, 0.20, 0.30, 0.50]:
    m = max(C, int(round(bs * n_tr)))
    for B in [10, 25]:
        stored_gb = B * C * m**2 * 8 / 1e9
        if require_gb(stored_gb + 1.0, f'bag={bs} B={B}'):
            t2.add_row([f'{bs:.2f}', B, f'{m}/{n_tr}', f'{stored_gb:.3f}', 'SKIP', '-'])
            continue
        t0 = time.time()
        bag = BaggingKLRClassifier(
            base_estimator=KLRClassifier(ridge=1e-3),
            n_estimators=B, bag_size=bs,
            n_jobs=1,          # <-- serial: never more than 1 bag in memory at once
            random_state=0,
        ).fit(X_tr, y_tr)
        elapsed = time.time() - t0
        t2.add_row([f'{bs:.2f}', B, f'{m}/{n_tr}', f'{stored_gb:.3f}',
                    f'{bag.score(X_te2, y_te2):.3f}', f'{elapsed:.1f}s'])
        del bag; gc.collect()

print(t2)

print('\nFeature subsampling (column pruning)  |  bag_size=0.3  B=20:')
t3 = PrettyTable(['feat_frac', 'Test', 'Time'])
for ff in [0.1, 0.25, 0.5, 0.75, None]:
    m = max(C, int(round(0.3 * n_tr)))
    stored_gb = 20 * C * m**2 * 8 / 1e9
    if require_gb(stored_gb + 1.0, f'ff={ff}'):
        t3.add_row([str(ff), 'SKIP', '-'])
        continue
    t0 = time.time()
    bag = BaggingKLRClassifier(
        base_estimator=KLRClassifier(ridge=1e-3),
        n_estimators=20, bag_size=0.3, feature_fraction=ff,
        n_jobs=1, random_state=0,
    ).fit(X_tr, y_tr)
    elapsed = time.time() - t0
    t3.add_row([str(ff), f'{bag.score(X_te2, y_te2):.3f}', f'{elapsed:.1f}s'])
    del bag; gc.collect()
print(t3)
