"""
bench_cifar.py
==============
Benchmarks KLRClassifier and BaggingKLRClassifier on CIFAR-10 (raw pixels,
32×32×3 = 3072 features, normalised to [0, 1]).

Note: raw-pixel KLR is not competitive with CNN-based methods; the goal here
is to study computational behaviour, regularisation sensitivity, and
spectral-pruning via bagging at moderate scale.

Memory accounting and guards are identical to bench_mnist.py.

Usage
-----
    cd <project-root>
    python3 tests/bench_cifar.py | tee tests/out/bench_cifar.txt
"""

import sys, os, gc, time, warnings, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import numpy as np
import psutil
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.classification import KLRClassifier, BaggingKLRClassifier, NystromKLRClassifier

CIFAR_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'cifar-10-batches-py')
N_CLASSES  = 10
HEADROOM   = 4.0


# ---------------------------------------------------------------------------
# Helpers (same as bench_mnist.py)
# ---------------------------------------------------------------------------

def mem_free_gb():
    return psutil.virtual_memory().available / 1e9

def klr_footprint_gb(n: int, C: int = N_CLASSES) -> float:
    return C * n**2 * 8 / 1e9

def bag_footprint_gb(n_train: int, bag_size, B: int, C: int = N_CLASSES) -> float:
    m = int(round(bag_size * n_train)) if isinstance(bag_size, float) else bag_size
    return B * C * m**2 * 8 / 1e9

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
# Load CIFAR-10
# ---------------------------------------------------------------------------

def load_cifar10(data_dir: str):
    def unpickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    Xs, ys = [], []
    for i in range(1, 6):
        d = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        Xs.append(d[b'data'].astype(np.float32))
        ys.extend(d[b'labels'])
    d = unpickle(os.path.join(data_dir, 'test_batch'))
    Xs.append(d[b'data'].astype(np.float32))
    ys.extend(d[b'labels'])
    return np.vstack(Xs) / 255.0, np.array(ys, dtype=np.int32)


section('Loading CIFAR-10')
X_full, y_full = load_cifar10(CIFAR_DIR)
print(f'  shape={X_full.shape}  |  classes={np.unique(y_full)}')

mem = psutil.virtual_memory()
print(f'  RAM: {mem.total/1e9:.1f} GB total, {mem_free_gb():.1f} GB free')

n_train = safe_n_train()
n_test  = 1000
print(f'  Memory cap -> n_train = {n_train}  '
      f'(footprint {klr_footprint_gb(n_train):.2f} GB)')

rng = np.random.RandomState(0)
tr_idx, te_idx = [], []
for c in range(N_CLASSES):
    ci = np.where(y_full == c)[0]; rng.shuffle(ci)
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

section('SINGLE KLR  |  CIFAR-10  (raw pixels)')
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

section('REGULARISATION PATH  |  CIFAR-10  |  band_factor=1.0')
t2 = PrettyTable(['ridge', 'Train', 'Test'])
for r in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    fp = klr_footprint_gb(len(X_tr2))
    if require_gb(fp, f'r={r:.0e}'): t2.add_row([f'{r:.0e}', '-', '-']); continue
    clf = KLRClassifier(ridge=r, band_factor=1.0).fit(X_tr2, y_tr2)
    t2.add_row([f'{r:.0e}',
                f'{clf.score(X_tr2, y_tr2):.3f}',
                f'{clf.score(X_te,  y_te ):.3f}'])
    del clf; gc.collect()
print(t2)


# ---------------------------------------------------------------------------
# 3. Bagging — spectral pruning
# ---------------------------------------------------------------------------

section('BAGGING KLR  |  CIFAR-10  |  n_jobs=1')
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


# ---------------------------------------------------------------------------
# 4. Nyström KLR — large-n regime
# ---------------------------------------------------------------------------
# Nyström footprint: C * m^2 * 8  bytes  (Cholesky on rank-m space)
#                  + m * d * 8    bytes  (landmarks, d = 3072)
# Peak during fit:  n_train * m * 8 bytes for K_nm block (freed immediately).
#
# This section runs on the FULL training set (50 000 samples) where full KLR
# would require C*n^2*8 ≈ 10*50000^2*8 = 200 GB — completely infeasible.
# Nyström with m=1000 needs only ~0.086 GB persistent + ~1.5 GB peak K_nm.

section('NYSTROM KLR  |  CIFAR-10  (raw pixels)  |  large-n regime')
print('  Full KLR would need C·n²·8 bytes — infeasible at n=50000.')
print('  Nyström replaces n×n kernel with n×m feature matrix.\n')

# Reload full training set for large-n experiments
import pickle as _pk

def _load_cifar_full(data_dir):
    Xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as fh:
            d = _pk.load(fh, encoding='bytes')
        Xs.append(d[b'data'].astype(np.float32))
        ys.extend(d[b'labels'])
    return np.vstack(Xs) / 255.0, np.array(ys, dtype=np.int32)

X_big, y_big = _load_cifar_full(CIFAR_DIR)
print(f'  Full train loaded: {X_big.shape}')
d_feat = X_big.shape[1]   # 3072

def nystrom_footprint_gb(m, d=d_feat, C=N_CLASSES):
    return (C * m**2 + m * d) * 8 / 1e9

def nystrom_peak_gb(n, m):
    """Peak during fit: K_nm block."""
    return n * m * 8 / 1e9

t4 = PrettyTable(['n_train', 'm', 'rank', 'Persist. GB', 'Peak GB',
                  'Full-KLR GB', 'Test', 'Time (s)'])

for n_large, m_land in [
    (10_000,  500),
    (10_000, 1000),
    (50_000,  500),
    (50_000, 1000),
    (50_000, 2000),
]:
    fp_ny   = nystrom_footprint_gb(m_land)
    pk_ny   = nystrom_peak_gb(n_large, m_land)
    fp_full = N_CLASSES * n_large**2 * 8 / 1e9

    needed = fp_ny + pk_ny
    if require_gb(needed, f'n={n_large} m={m_land}'):
        t4.add_row([n_large, m_land, '-',
                    f'{fp_ny:.3f}', f'{pk_ny:.2f}', f'{fp_full:.1f}',
                    'SKIP', '-'])
        continue

    # stratified subsample of full training set
    rng_big = np.random.RandomState(42)
    idx_big = []
    for c in range(N_CLASSES):
        ci = np.where(y_big == c)[0]
        rng_big.shuffle(ci)
        idx_big.extend(ci[: n_large // N_CLASSES].tolist())
    Xn = X_big[idx_big]; yn = y_big[idx_big]

    t0 = time.time()
    clf = NystromKLRClassifier(
        ridge=1e-3, band_factor=1.0,
        n_landmarks=m_land, random_state=0,
    ).fit(Xn, yn)
    elapsed = time.time() - t0
    t4.add_row([n_large, m_land, clf.nystrom_rank_,
                f'{fp_ny:.3f}', f'{pk_ny:.2f}', f'{fp_full:.1f}',
                f'{clf.score(X_te, y_te):.3f}', f'{elapsed:.1f}'])
    del clf, Xn, yn; gc.collect()

del X_big, y_big; gc.collect()
print(t4)

# ---------------------------------------------------------------------------
# 5. Tuned Nyström on CIFAR-10
# ---------------------------------------------------------------------------

section('TUNED NYSTROM  |  CIFAR-10  |  m=1000  (grid: ridge × band_factor)')
print('  Tuning cost: O(nm) distances once, then O(m³C·|grid|) per config.\n')

# Reload a fixed-size train set for fair comparison with Sections 1–3
X_tune_tr, X_tune_val = X_tr2, X_val

m_tune = 1000
fp_tune = nystrom_footprint_gb(m_tune)
if not require_gb(fp_tune + nystrom_peak_gb(len(X_tune_tr), m_tune) + 0.5,
                  'TunedNystrom'):
    t0 = time.time()
    clf = NystromKLRClassifier(n_landmarks=m_tune, random_state=0).tune(
        X_tune_tr, y_tr2, X_tune_val, y_val, verbose=True)
    elapsed = time.time() - t0
    t5 = PrettyTable(['Method', 'best_ridge', 'best_band_factor', 'Val', 'Test', 'Time (s)'])
    t5.add_row([f'Nyström m={m_tune} (tuned)',
                f'{clf.ridge:.2e}', f'{clf.band_factor}',
                f'{clf.score(X_tune_val, y_val):.3f}',
                f'{clf.score(X_te, y_te):.3f}',
                f'{elapsed:.1f}'])
    print(t5)
    del clf; gc.collect()
else:
    print('  Skipped (insufficient memory).')
