"""
plot_ridge_tradeoff.py
======================
Visualises the overfitting / stability trade-off controlled by the ridge γ
for KLRClassifier on multiple datasets.

Layout
------
Row per dataset (4 synthetic 2-D + MNIST):
  Cols 0-2: Decision boundaries at γ = high / optimal / low  [2-D only]
  Col  3  : Regularisation path (train / val / test accuracy vs log γ)
  For MNIST the boundary panels are replaced by a single wide accuracy panel.

Outputs
-------
    tests/out/images/ridge_tradeoff.png

Usage
-----
    cd <project-root>
    python3 tests/plot_ridge_tradeoff.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, fetch_openml
from sklearn.preprocessing import LabelEncoder

from src.classification import KLRClassifier
from tests.utils import make4moons, make4circles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE  = 0
RESOLUTION    = 110
RIDGE_FINE    = np.logspace(-5, 1, 35)
N_SYNTH       = 300           # samples per 2-D dataset
N_MNIST_TRAIN = 2000          # small enough to sweep quickly
OUT_PATH      = os.path.join(os.path.dirname(__file__), 'out', 'images',
                              'ridge_tradeoff.png')

# Per-class colour scheme (scatter hex, contourf cmap)
CLASS_STYLES = [
    ('#d62728', plt.cm.Reds),
    ('#1f77b4', plt.cm.Blues),
    ('#2ca02c', plt.cm.Greens),
    ('#ff7f0e', plt.cm.Oranges),
]

def _sc_colors(y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return [CLASS_STYLES[idx[yi]][0] for yi in y]

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

print('Building datasets…')

_rng = np.random.RandomState(RANDOM_STATE)

SYNTH_DATASETS = [
    ('Two moons',    *make_moons(  n_samples=N_SYNTH, noise=0.25, random_state=RANDOM_STATE)),
    ('Two circles',  *make_circles(n_samples=N_SYNTH, noise=0.12, factor=0.5, random_state=RANDOM_STATE)),
    ('Four moons',   *make4moons(  n_samples=150, noise=0.15, random_state=RANDOM_STATE)),
    ('Four circles', *make4circles(n_samples=150, noise=0.10, random_state=RANDOM_STATE)),
]

# MNIST subset
print('Loading MNIST…')
mnist  = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_mn   = mnist.data.astype(np.float32)
y_mn   = LabelEncoder().fit_transform(mnist.target)
# stratified subsample
idx_mn = []
for c in range(10):
    ci = np.where(y_mn == c)[0]; _rng.shuffle(ci)
    idx_mn.extend(ci[:N_MNIST_TRAIN // 10].tolist())
X_mn, y_mn = X_mn[idx_mn], y_mn[idx_mn]
del mnist

# ---------------------------------------------------------------------------
# Helper: run regularisation path
# ---------------------------------------------------------------------------

def reg_path(X, y, band_factor=1.0):
    """Return (ridges, tr, val, te, best_ridge) for ridge sweep."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    X_val, X_te2, y_val, y_te2 = train_test_split(
        X_te, y_te, test_size=0.50, random_state=RANDOM_STATE)

    tr, val, te = [], [], []
    for r in RIDGE_FINE:
        clf = KLRClassifier(ridge=r, band_factor=band_factor).fit(X_tr, y_tr)
        tr.append(clf.score(X_tr, y_tr))
        val.append(clf.score(X_val, y_val))
        te.append(clf.score(X_te2, y_te2))

    best_ridge = RIDGE_FINE[int(np.argmax(val))]
    return X_tr, y_tr, X_val, y_val, X_te2, y_te2, \
           np.array(tr), np.array(val), np.array(te), best_ridge


def draw_boundary(ax, clf, X_tr, y_tr, X_te, y_te,
                  x0, x1, y0, y1):
    """Fill decision regions and scatter train/test points."""
    xx, yy = np.meshgrid(np.linspace(x0, x1, RESOLUTION),
                         np.linspace(y0, y1, RESOLUTION))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    proba  = clf.predict_proba(grid)
    y_pred = clf.predict(grid).reshape(xx.shape)
    classes = clf.classes_

    for i, c in enumerate(classes):
        _, cmap = CLASS_STYLES[i]
        col_idx = np.searchsorted(classes, c)
        prob_c  = proba[:, col_idx].reshape(xx.shape)
        ax.contourf(xx, yy, np.where(y_pred == c, prob_c, np.nan),
                    cmap=cmap, alpha=0.70, levels=np.linspace(0.4, 1.0, 6))

    ax.contour(xx, yy, y_pred, colors='k', linewidths=0.7, alpha=0.35)
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=_sc_colors(y_tr, classes),
               edgecolors='k', linewidths=0.4, s=16, zorder=3)
    ax.scatter(X_te[:, 0], X_te[:, 1], c=_sc_colors(y_te, classes),
               edgecolors='k', linewidths=0.4, s=16, alpha=0.4, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])


def draw_acc_curve(ax, ridges, tr, val, te, best_ridge, ds_name, y_lo=None, y_hi=None):
    """Draw train/val/test accuracy vs log10(ridge)."""
    log_r = np.log10(ridges)
    ax.plot(log_r, tr,  color='#2ca02c', lw=1.5, ls='--', label='Train')
    ax.plot(log_r, val, color='#ff7f0e', lw=1.8, label='Val')
    ax.plot(log_r, te,  color='#1f77b4', lw=1.5, label='Test')

    best_log = np.log10(best_ridge)
    ax.axvspan(best_log - 0.5, best_log + 0.5, color='#2ca02c', alpha=0.10)
    ax.axvline(best_log, color='#2ca02c', lw=1.0, ls=':')

    # mark selected ridges from decision-boundary panels
    for r, style in [(RIDGE_FINE[-1],        ('v', '#4393c3')),  # high → underfit
                     (best_ridge,             ('D', '#2ca02c')),  # optimal
                     (RIDGE_FINE[0],          ('^', '#d73027'))]: # low  → overfit
        vi = np.argmin(np.abs(ridges - r))
        ax.scatter([np.log10(r)], [val[vi]], marker=style[0], color=style[1],
                   s=50, zorder=5)

    ax.set_xlabel(r'$\log_{10}(\gamma)$', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.set_xlim(log_r.min(), log_r.max())
    lo = (y_lo or 0.45); hi = (y_hi or 1.01)
    ax.set_ylim(lo, hi)
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.85)
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.set_title(f'{ds_name} — reg. path', fontsize=8.5)


# ---------------------------------------------------------------------------
# Pre-compute everything
# ---------------------------------------------------------------------------

print('Computing regularisation paths for 2-D datasets…')
synth_results = []
for ds_name, X, y in SYNTH_DATASETS:
    print(f'  {ds_name}…', end=' ', flush=True)
    X_tr, y_tr, X_val, y_val, X_te, y_te, tr, val, te, best_ridge = reg_path(X, y)
    synth_results.append((ds_name, X, y, X_tr, y_tr, X_val, y_val,
                          X_te, y_te, tr, val, te, best_ridge))
    print('done')

print('Computing regularisation path for MNIST…')
(X_mn_tr, y_mn_tr, X_mn_val, y_mn_val,
 X_mn_te, y_mn_te, tr_mn, val_mn, te_mn, best_ridge_mn) = reg_path(X_mn, y_mn)
print('done')


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

n_synth = len(SYNTH_DATASETS)
# 5 rows (4 synth + 1 MNIST), 4 columns (3 boundary + 1 acc)
# For MNIST row: boundary columns are collapsed into the acc panel
fig = plt.figure(figsize=(14, 3.2 * (n_synth + 1)))
gs  = gridspec.GridSpec(
    n_synth + 1, 4,
    width_ratios=[1, 1, 1, 1.6],
    hspace=0.30, wspace=0.07,
)

PANEL_RIDGES = [RIDGE_FINE[-1], None, RIDGE_FINE[0]]  # high, best, low
PANEL_LABELS = [r'$\gamma$ high (underfit)', r'$\gamma$ optimal', r'$\gamma$ low (overfit)']
BORDER_COLORS = ['#4393c3', '#2ca02c', '#d73027']

for row, (ds_name, X, y, X_tr, y_tr, X_val, y_val,
          X_te, y_te, tr, val, te, best_ridge) in enumerate(synth_results):

    classes = np.unique(y)
    pad = 0.4
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad

    # three boundary panels
    for col, (ridge_sel, plabel, bcol) in enumerate(
            zip(PANEL_RIDGES, PANEL_LABELS, BORDER_COLORS)):
        ax = fig.add_subplot(gs[row, col])
        r  = best_ridge if ridge_sel is None else ridge_sel
        clf = KLRClassifier(ridge=r, band_factor=1.0).fit(X_tr, y_tr)
        tr_acc = clf.score(X_tr, y_tr)
        te_acc = clf.score(X_te, y_te)

        draw_boundary(ax, clf, X_tr, y_tr, X_te, y_te, x0, x1, y0, y1)

        kw = dict(transform=ax.transAxes, ha='right',
                  path_effects=[pe.withStroke(linewidth=2, foreground='w')])
        ax.text(0.97, 0.08, f'{te_acc:.2f}', fontsize=10, color='k', **kw)
        ax.text(0.97, 0.02, f'{tr_acc:.2f}', fontsize=8,  color='0.4', **kw)

        for sp in ax.spines.values():
            sp.set_color(bcol); sp.set_linewidth(2.0)

        if row == 0:
            ax.set_title(plabel, fontsize=8.5, color=bcol)
        if col == 0:
            ax.set_ylabel(ds_name, fontsize=9, labelpad=3)

    # accuracy curve
    ax_acc = fig.add_subplot(gs[row, 3])
    draw_acc_curve(ax_acc, RIDGE_FINE, tr, val, te, best_ridge, ds_name)
    if row == 0:
        ax_acc.set_title('Reg. path', fontsize=8.5)

# MNIST row: span all 4 cols
ax_mn = fig.add_subplot(gs[n_synth, :])
draw_acc_curve(ax_mn, RIDGE_FINE, tr_mn, val_mn, te_mn, best_ridge_mn,
               f'MNIST (n={N_MNIST_TRAIN})',
               y_lo=0.3, y_hi=1.01)
ax_mn.set_title(
    f'MNIST-784  n_train={N_MNIST_TRAIN}  — regularisation path  '
    r'(band\_factor = 1.0)',
    fontsize=9,
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved → {OUT_PATH}')
