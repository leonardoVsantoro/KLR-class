"""
bench_mnist.py
==============
Benchmarks KLRClassifier and BaggingKLRClassifier on MNIST-784.

Memory accounting
-----------------
A fitted KLRClassifier on n training samples with C classes stores:

    C * n^2 * 8  bytes  (Cholesky factors, store_covariance=False default)

plus n*d*8 bytes for X_train_ and C*n*8 for means_ (negligible at d=784).
For MNIST with C=10:

    n=1000 -> 0.08 GB      n=2000 -> 0.32 GB
    n=3000 -> 0.72 GB      n=4000 -> 1.28 GB

This script queries available RAM at startup, then caps n_train so that
the single-KLR persistent footprint leaves 4 GB headroom.  All BaggingKLR
runs use n_jobs=1 and check memory before starting.

Usage
-----
    cd <project-root>
    python3 tests/bench_mnist.py | tee tests/out/bench_mnist.txt
"""

import sys, os, gc, time, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import numpy as np
import psutil
from prettytable import PrettyTable
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.classification import KLRClassifier, BaggingKLRClassifier

N_CLASSES  = 10
HEADROOM   = 4.0   # GB to keep free at all times


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def mem_free_gb():
    return psutil.virtual_memory().available / 1e9

def klr_footprint_gb(n: int, C: int = N_CLASSES) -> float:
    """Persistent GB for a fitted KLRClassifier (store_covariance=False)."""
    return C * n**2 * 8 / 1e9

def bag_footprint_gb(n_train: int, bag_size, B: int, C: int = N_CLASSES) -> float:
    m = int(round(bag_size * n_train)) if isinstance(bag_size, float) else bag_size
    return B * C * m**2 * 8 / 1e9

def safe_n_train(C: int = N_CLASSES, headroom: float = HEADROOM,
                 n_max: int = 5000) -> int:
    """Largest n such that klr_footprint_gb(n, C) + headroom <= mem_free."""
    budget = max(0.0, mem_free_gb() - headroom)
    n = int(np.sqrt(budget * 1e9 / (C * 8)))
    return min(n, n_max)

def require_gb(needed: float, label: str = '') -> bool:
    free = mem_free_gb()
    if free - needed < HEADROOM:
        tag = f' [{label}]' if label else ''
        print(f'  SKIP{tag}: {free:.1f} GB free, need {needed:.1f} + {HEADROOM:.0f} GB headroom')
        return True
    return False

def section(title: str):
    print(f'\n{"="*65}')
    print(title)
    print('='*65)


# ---------------------------------------------------------------------------
# Load MNIST
# ---------------------------------------------------------------------------

section('Loading MNIST-784')
t0 = time.time()
mnist    = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_full   = mnist.data.astype(np.float32)
y_full   = LabelEncoder().fit_transform(mnist.target)
print(f'  Loaded in {time.time()-t0:.1f}s  |  shape={X_full.shape}')

mem = psutil.virtual_memory()
print(f'  RAM: {mem.total/1e9:.1f} GB total, {mem_free_gb():.1f} GB free')

n_train = safe_n_train()
n_test  = 1000
print(f'  Memory cap -> n_train = {n_train}  '
      f'(footprint {klr_footprint_gb(n_train):.2f} GB)')

# Stratified subsample
rng = np.random.RandomState(42)
tr_idx, te_idx = [], []
for c in range(N_CLASSES):
    ci = np.where(y_full == c)[0]
    rng.shuffle(ci)
    per_tr = n_train // N_CLASSES
    per_te = n_test  // N_CLASSES
    tr_idx.extend(ci[:per_tr])
    te_idx.extend(ci[per_tr: per_tr + per_te])

X_tr = X_full[tr_idx]; y_tr = y_full[tr_idx]
X_te = X_full[te_idx]; y_te = y_full[te_idx]
del X_full, y_full; gc.collect()

X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr, y_tr, test_size=0.2, random_state=0)
print(f'  Train={len(X_tr2)}  Val={len(X_val)}  Test={len(X_te)}')


# ---------------------------------------------------------------------------
# 1. Single KLR
# ---------------------------------------------------------------------------

section('SINGLE KLR  |  MNIST')
t = PrettyTable(['Model', 'Train', 'Test', 'Time (s)'])

for label, kwargs in [
    ('KLR (ridge=1e-1)', dict(ridge=1e-1, band_factor=1.0)),
    ('KLR (ridge=1e-2)', dict(ridge=1e-2, band_factor=1.0)),
    ('KLR (ridge=1e-3)', dict(ridge=1e-3, band_factor=1.0)),
    ('KLR (tuned)',      dict()),
]:
    fp = klr_footprint_gb(len(X_tr2))
    if require_gb(fp, label):
        t.add_row([label, '-', '-', '-']); continue
    t0 = time.time()
    if label == 'KLR (tuned)':
        clf = KLRClassifier().tune(X_tr2, y_tr2, X_val, y_val, verbose=True)
    else:
        clf = KLRClassifier(**kwargs).fit(X_tr2, y_tr2)
    elapsed = time.time() - t0
    t.add_row([label,
               f'{clf.score(X_tr2, y_tr2):.3f}',
               f'{clf.score(X_te,  y_te ):.3f}',
               f'{elapsed:.1f}'])
    del clf; gc.collect()

# Baselines
for name, clf in [('KNN (k=5)', KNeighborsClassifier(5)),
                  ('Random Forest', RandomForestClassifier(100, random_state=0))]:
    t0 = time.time()
    clf.fit(X_tr2, y_tr2)
    t.add_row([name,
               f'{clf.score(X_tr2, y_tr2):.3f}',
               f'{clf.score(X_te,  y_te ):.3f}',
               f'{time.time()-t0:.1f}'])
    del clf; gc.collect()
print(t)


# ---------------------------------------------------------------------------
# 2. Regularisation path
# ---------------------------------------------------------------------------

section('REGULARISATION PATH  |  MNIST  |  band_factor=1.0')
t2 = PrettyTable(['ridge', 'Train', 'Test', 'eff_ridge'])
for r in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    fp = klr_footprint_gb(len(X_tr2))
    if require_gb(fp, f'r={r:.0e}'): t2.add_row([f'{r:.0e}','-','-','-']); continue
    clf = KLRClassifier(ridge=r, band_factor=1.0).fit(X_tr2, y_tr2)
    t2.add_row([f'{r:.0e}',
                f'{clf.score(X_tr2, y_tr2):.3f}',
                f'{clf.score(X_te,  y_te ):.3f}',
                f'{clf.effective_ridge_:.3e}'])
    del clf; gc.collect()
print(t2)


# ---------------------------------------------------------------------------
# 3. Bagging — spectral pruning
# ---------------------------------------------------------------------------

section('BAGGING KLR  |  MNIST  |  n_jobs=1  spectral pruning')
print('  Each bag is a stratified subsample of X_tr.')
print('  Persistent mem = B * C * m^2 * 8 bytes.\n')

t3 = PrettyTable(['bag_size', 'n_est', 'm/bag', 'stored_GB', 'Test', 'Time'])
for bs, B in [(0.10, 20), (0.15, 20), (0.20, 15), (0.30, 10)]:
    fp = bag_footprint_gb(len(X_tr), bs, B)
    if require_gb(fp, f'bag={bs} B={B}'):
        m = int(round(bs * len(X_tr)))
        t3.add_row([f'{bs:.2f}', B, m, f'{fp:.3f}', 'SKIP', '-']); continue
    t0 = time.time()
    bag = BaggingKLRClassifier(
        base_estimator=KLRClassifier(ridge=1e-3, band_factor=1.0),
        n_estimators=B, bag_size=bs,
        n_jobs=1, random_state=0,
    ).fit(X_tr, y_tr)
    elapsed = time.time() - t0
    m = int(round(bs * len(X_tr)))
    t3.add_row([f'{bs:.2f}', B, m, f'{fp:.3f}',
                f'{bag.score(X_te, y_te):.3f}', f'{elapsed:.1f}s'])
    del bag; gc.collect()
print(t3)
