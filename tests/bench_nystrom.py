"""
bench_nystrom.py
================
Benchmarks NystromKLRClassifier against full KLRClassifier on MNIST-784,
illustrating the accuracy vs. n_landmarks trade-off and memory savings.

Memory footprint
----------------
Full KLR:    C · n² · 8  bytes  (Cholesky factors, n = training set size)
Nyström KLR: C · m² · 8  bytes  (Cholesky factors, m = n_landmarks)
             + m · d · 8  bytes  (landmarks, d = 784)

For C=10, n=5000, d=784:
    Full KLR     : 5000²  · 10 · 8 = 2.00 GB
    Nyström m=500: 500²   · 10 · 8 = 0.02 GB  (100× smaller)
    Nyström m=200: 200²   · 10 · 8 = 0.003 GB

Peak memory during fit is O(nm) for the K_nm distance block (freed immediately).

Usage
-----
    cd <project-root>
    python3 tests/bench_nystrom.py | tee tests/out/bench_nystrom.txt
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

from src.classification import KLRClassifier, NystromKLRClassifier

N_CLASSES = 10
HEADROOM  = 4.0    # GB headroom at all times


def mem_free_gb():
    return psutil.virtual_memory().available / 1e9

def klr_footprint_gb(n: int, C: int = N_CLASSES) -> float:
    return C * n**2 * 8 / 1e9

def nystrom_footprint_gb(m: int, d: int, C: int = N_CLASSES) -> float:
    return (C * m**2 + m * d) * 8 / 1e9

def safe_n_train(C: int = N_CLASSES, headroom: float = HEADROOM,
                 n_max: int = 5000) -> int:
    budget = max(0.0, mem_free_gb() - headroom)
    return min(int(np.sqrt(budget * 1e9 / (C * 8))), n_max)

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
mnist  = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_full = mnist.data.astype(np.float32)
y_full = LabelEncoder().fit_transform(mnist.target)
print(f'  Loaded in {time.time()-t0:.1f}s  |  shape={X_full.shape}')

mem = psutil.virtual_memory()
print(f'  RAM: {mem.total/1e9:.1f} GB total, {mem_free_gb():.1f} GB free')

n_train = safe_n_train()
n_test  = 2000
print(f'  Memory cap -> n_train = {n_train}  '
      f'(full-KLR footprint {klr_footprint_gb(n_train):.2f} GB)')

rng = np.random.RandomState(42)
tr_idx, te_idx = [], []
for c in range(N_CLASSES):
    ci = np.where(y_full == c)[0]
    rng.shuffle(ci)
    per_tr = n_train  // N_CLASSES
    per_te = n_test   // N_CLASSES
    tr_idx.extend(ci[:per_tr])
    te_idx.extend(ci[per_tr: per_tr + per_te])

X_tr = X_full[tr_idx]; y_tr = y_full[tr_idx]
X_te = X_full[te_idx]; y_te = y_full[te_idx]
del X_full, y_full; gc.collect()

X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr, y_tr, test_size=0.2, random_state=0)
print(f'  Train={len(X_tr2)}  Val={len(X_val)}  Test={len(X_te)}')


# ---------------------------------------------------------------------------
# 1. Accuracy vs. n_landmarks table
# ---------------------------------------------------------------------------

section('NYSTROM vs FULL KLR  |  MNIST  |  ridge=1e-3  band_factor=1.0')
t = PrettyTable(['Method', 'n_landmarks', 'rank', 'Footprint (GB)',
                 'Train', 'Test', 'Time (s)'])

# Baseline: full KLR
fp_full = klr_footprint_gb(len(X_tr2))
if not require_gb(fp_full, 'FullKLR'):
    t0 = time.time()
    clf = KLRClassifier(ridge=1e-3, band_factor=1.0).fit(X_tr2, y_tr2)
    elapsed = time.time() - t0
    t.add_row(['FullKLR', len(X_tr2), len(X_tr2),
               f'{fp_full:.3f}',
               f'{clf.score(X_tr2, y_tr2):.4f}',
               f'{clf.score(X_te,  y_te ):.4f}',
               f'{elapsed:.1f}'])
    del clf; gc.collect()
else:
    t.add_row(['FullKLR', len(X_tr2), '-', f'{fp_full:.3f}', 'SKIP', 'SKIP', '-'])

d = X_tr2.shape[1]
for m in [50, 100, 200, 500, 1000, 2000]:
    if m > len(X_tr2):
        continue
    fp = nystrom_footprint_gb(m, d)
    if require_gb(fp + 0.5, f'm={m}'):
        t.add_row([f'Nyström', m, '-', f'{fp:.4f}', 'SKIP', 'SKIP', '-'])
        continue
    t0 = time.time()
    clf = NystromKLRClassifier(ridge=1e-3, band_factor=1.0,
                               n_landmarks=m, random_state=0).fit(X_tr2, y_tr2)
    elapsed = time.time() - t0
    t.add_row(['Nyström', m, clf.nystrom_rank_,
               f'{fp:.4f}',
               f'{clf.score(X_tr2, y_tr2):.4f}',
               f'{clf.score(X_te,  y_te ):.4f}',
               f'{elapsed:.1f}'])
    del clf; gc.collect()

print(t)


# ---------------------------------------------------------------------------
# 2. Regularisation path for Nyström  (m=500)
# ---------------------------------------------------------------------------

section('REGULARISATION PATH  |  Nyström m=500  |  MNIST  |  band_factor=1.0')
t2 = PrettyTable(['ridge', 'Train', 'Val', 'Test', 'effective_ridge'])
m_fixed = min(500, len(X_tr2))
for r in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    fp = nystrom_footprint_gb(m_fixed, d)
    if require_gb(fp + 0.5, f'r={r:.0e}'):
        t2.add_row([f'{r:.0e}', '-', '-', '-', '-']); continue
    clf = NystromKLRClassifier(ridge=r, band_factor=1.0,
                               n_landmarks=m_fixed, random_state=0).fit(X_tr2, y_tr2)
    t2.add_row([f'{r:.0e}',
                f'{clf.score(X_tr2, y_tr2):.4f}',
                f'{clf.score(X_val, y_val):.4f}',
                f'{clf.score(X_te,  y_te ):.4f}',
                f'{clf.effective_ridge_:.3e}'])
    del clf; gc.collect()
print(t2)


# ---------------------------------------------------------------------------
# 3. Tuned Nyström vs. tuned full KLR
# ---------------------------------------------------------------------------

section('TUNED COMPARISON  |  MNIST  |  Nyström m=500 vs FullKLR')
t3 = PrettyTable(['Method', 'best_ridge', 'best_band_factor', 'Val', 'Test', 'Time (s)'])

# Tuned full KLR
fp_full = klr_footprint_gb(len(X_tr2))
if not require_gb(fp_full, 'TunedFullKLR'):
    t0 = time.time()
    clf = KLRClassifier().tune(X_tr2, y_tr2, X_val, y_val, verbose=True)
    elapsed = time.time() - t0
    t3.add_row(['FullKLR (tuned)',
                f'{clf.ridge:.2e}', f'{clf.band_factor}',
                f'{clf.score(X_val, y_val):.4f}',
                f'{clf.score(X_te,  y_te ):.4f}',
                f'{elapsed:.1f}'])
    del clf; gc.collect()
else:
    t3.add_row(['FullKLR (tuned)', '-', '-', 'SKIP', 'SKIP', '-'])

# Tuned Nyström
fp_ny = nystrom_footprint_gb(m_fixed, d)
if not require_gb(fp_ny + 0.5, 'TunedNystrom'):
    t0 = time.time()
    clf = NystromKLRClassifier(n_landmarks=m_fixed, random_state=0).tune(
        X_tr2, y_tr2, X_val, y_val, verbose=True)
    elapsed = time.time() - t0
    t3.add_row([f'Nyström m={m_fixed} (tuned)',
                f'{clf.ridge:.2e}', f'{clf.band_factor}',
                f'{clf.score(X_val, y_val):.4f}',
                f'{clf.score(X_te,  y_te ):.4f}',
                f'{elapsed:.1f}'])
    del clf; gc.collect()
else:
    t3.add_row([f'Nyström m={m_fixed} (tuned)', '-', '-', 'SKIP', 'SKIP', '-'])

print(t3)


# ---------------------------------------------------------------------------
# 4. Nyström on large n (5×–10× more data than safe_n_train for full KLR)
# ---------------------------------------------------------------------------

section('NYSTROM ON LARGE n  |  MNIST  |  m=1000 fixed  (full KLR would OOM)')
print('  Full KLR Cholesky footprint at n:')

t4 = PrettyTable(['n_train', 'Full-KLR GB', 'Nyström GB (m=1000)', 'Test', 'Time (s)'])
m_large = 1000
fp_ny_large = nystrom_footprint_gb(m_large, d)

# Build a larger training set from MNIST (reload)
t0_load = time.time()
mnist2  = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_big   = mnist2.data.astype(np.float32)
y_big   = LabelEncoder().fit_transform(mnist2.target)
del mnist2; gc.collect()

for n_large in [10000, 20000, 50000]:
    fp_full_large = klr_footprint_gb(n_large)
    rng2 = np.random.RandomState(7)
    idx2 = []
    for c in range(N_CLASSES):
        ci = np.where(y_big == c)[0]
        rng2.shuffle(ci)
        idx2.extend(ci[: n_large // N_CLASSES].tolist())
    Xn = X_big[idx2]; yn = y_big[idx2]
    del idx2

    if require_gb(fp_ny_large + 0.5, f'n={n_large}'):
        t4.add_row([n_large, f'{fp_full_large:.2f}', f'{fp_ny_large:.4f}', 'SKIP', '-'])
        del Xn, yn; gc.collect()
        continue

    t0 = time.time()
    clf = NystromKLRClassifier(ridge=1e-3, band_factor=1.0,
                               n_landmarks=m_large, random_state=0).fit(Xn, yn)
    elapsed = time.time() - t0
    t4.add_row([n_large, f'{fp_full_large:.2f}', f'{fp_ny_large:.4f}',
                f'{clf.score(X_te, y_te):.4f}', f'{elapsed:.1f}'])
    del clf, Xn, yn; gc.collect()

del X_big, y_big; gc.collect()
print(t4)
