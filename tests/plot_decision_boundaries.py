"""
plot_decision_boundaries.py
===========================
Compares KLRClassifier against standard sklearn classifiers on four synthetic
2-D datasets.  Each row is a dataset; the first column shows raw data with
KDE density contours; subsequent columns show per-class probability shading
with test accuracy (large) and train accuracy (small, faded) annotated.

Output
------
    tests/out/images/decision_boundaries.png

Usage
-----
    cd <project-root>
    python3 tests/plot_decision_boundaries.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.classification import KLRClassifier
from tests.utils import make_moons, make_circles, make4moons, make4circles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESOLUTION   = 120
TEST_SIZE    = 0.3
RANDOM_STATE = 42
OUT_PATH     = os.path.join(os.path.dirname(__file__), 'out', 'images',
                            'decision_boundaries.png')

# (label, X, y)
DATASETS = [
    ('Two moons',    *make_moons(  n_samples=300, noise=0.20, random_state=RANDOM_STATE)),
    ('Two circles',  *make_circles(n_samples=300, noise=0.12, factor=0.5,
                                   random_state=RANDOM_STATE)),
    ('Four moons',   *make4moons(  n_samples=150, noise=0.15, random_state=RANDOM_STATE)),
    ('Four circles', *make4circles(n_samples=150, noise=0.10, random_state=RANDOM_STATE)),
]

CLASSIFIERS = {
    'KLR (tuned)':   KLRClassifier(),
    'KLR (default)': KLRClassifier(ridge=1e-3, band_factor=1.0),
    'KNN (k=5)':     KNeighborsClassifier(5),
    'SVM (RBF)':     SVC(probability=True, random_state=RANDOM_STATE),
    'QDA':           QuadraticDiscriminantAnalysis(),
    'Random Forest': RandomForestClassifier(100, random_state=RANDOM_STATE),
    'MLP':           MLPClassifier(max_iter=600, random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------------------
# Per-class colour scheme
# Scatter colour for class j  ↔  contourf colormap for class j are aligned.
# ---------------------------------------------------------------------------

# (scatter_hex, contourf_cmap, kde_cmap_name)
CLASS_STYLES = [
    ('#d62728', plt.cm.Reds,    'Reds'),      # class 0 – red
    ('#1f77b4', plt.cm.Blues,   'Blues'),     # class 1 – blue
    ('#2ca02c', plt.cm.Greens,  'Greens'),    # class 2 – green
    ('#ff7f0e', plt.cm.Oranges, 'Oranges'),  # class 3 – orange
]

def scatter_colors(y, classes):
    """Return a list of hex colours matching CLASS_STYLES for each label."""
    idx_map = {c: i for i, c in enumerate(classes)}
    return [CLASS_STYLES[idx_map[yi]][0] for yi in y]

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

n_rows = len(DATASETS)
n_cols = 1 + len(CLASSIFIERS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(2.8 * n_cols, 2.9 * n_rows),
)
fig.subplots_adjust(wspace=0.05, hspace=0.14)

for row, (ds_name, X, y) in enumerate(DATASETS):
    classes   = np.unique(y)
    n_classes = len(classes)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_val, X_te2, y_val, y_te2 = train_test_split(
        X_te, y_te, test_size=0.5, random_state=RANDOM_STATE)

    pad = 0.4
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, RESOLUTION),
                         np.linspace(y0, y1, RESOLUTION))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # --- data column: KDE densities + scatter --------------------------------
    ax0 = axes[row, 0]
    for i, c in enumerate(classes):
        mask = y == c
        _, kde_cmap, kde_cmap_name = CLASS_STYLES[i]
        sns.kdeplot(x=X[mask, 0], y=X[mask, 1], ax=ax0,
                    fill=True, cmap=kde_cmap_name, bw_adjust=1.2,
                    thresh=0.05, alpha=0.55, levels=5)

    sc_colors_tr  = scatter_colors(y_tr,  classes)
    sc_colors_te2 = scatter_colors(y_te2, classes)
    ax0.scatter(X_tr[:,  0], X_tr[:,  1], c=sc_colors_tr,
                edgecolors='k', linewidths=0.4, s=20, zorder=3)
    ax0.scatter(X_te2[:, 0], X_te2[:, 1], c=sc_colors_te2,
                edgecolors='k', linewidths=0.4, s=20, alpha=0.4, zorder=3)
    ax0.set_xlim(x0, x1); ax0.set_ylim(y0, y1)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_ylabel(ds_name, fontsize=9, labelpad=4)
    if row == 0:
        ax0.set_title('Data + KDE', fontsize=9)

    # --- classifier columns --------------------------------------------------
    for col, (clf_name, clf_proto) in enumerate(CLASSIFIERS.items(), start=1):
        ax = axes[row, col]
        clf = clf_proto.__class__(**clf_proto.get_params())

        if hasattr(clf, 'tune'):
            clf = clf.tune(X_tr, y_tr, X_val, y_val, verbose=False)
        else:
            clf.fit(X_tr, y_tr)

        tr_acc = clf.score(X_tr,  y_tr)
        te_acc = clf.score(X_te2, y_te2)

        proba  = clf.predict_proba(grid)
        y_pred = clf.predict(grid).reshape(xx.shape)

        for i, c in enumerate(clf.classes_):
            _, region_cmap, _ = CLASS_STYLES[i]
            col_idx = np.searchsorted(clf.classes_, c)
            prob_c  = proba[:, col_idx].reshape(xx.shape)
            ax.contourf(xx, yy,
                        np.where(y_pred == c, prob_c, np.nan),
                        cmap=region_cmap, alpha=0.75,
                        levels=np.linspace(0.4, 1.0, 6))

        ax.contour(xx, yy, y_pred, colors='k', linewidths=0.5, alpha=0.35)

        # Scatter with matched colours
        ax_sc_tr  = scatter_colors(y_tr,  classes)
        ax_sc_te2 = scatter_colors(y_te2, classes)
        ax.scatter(X_tr[:,  0], X_tr[:,  1], c=ax_sc_tr,
                   edgecolors='k', linewidths=0.4, s=16, zorder=3)
        ax.scatter(X_te2[:, 0], X_te2[:, 1], c=ax_sc_te2,
                   edgecolors='k', linewidths=0.4, s=16, alpha=0.4, zorder=3)

        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])

        kw = dict(transform=ax.transAxes, ha='right',
                  path_effects=[pe.withStroke(linewidth=2, foreground='w')])
        ax.text(0.97, 0.08, f'{te_acc:.2f}', fontsize=11, color='k', **kw)
        ax.text(0.97, 0.02, f'{tr_acc:.2f}', fontsize=8,  color='0.4', **kw)

        if row == 0:
            ax.set_title(clf_name, fontsize=9)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved → {OUT_PATH}')
