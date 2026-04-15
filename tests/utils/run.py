import os
import sys

# Resolve project root regardless of working directory
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colormaps as mpl_colormaps
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.classification import KLRClassifier, BaggingKLRClassifier


def getAccScores(
    data,
    classifiers: dict,
    test_size: float = 0.2,
    cross_validate_KLR: bool = False,
    random_state: int = 42,
    verbose: bool = True,
    noise_scale: float = 0,
):
    """
    Evaluate classifiers on a dataset and print a results table.

    For classifiers that expose a ``tune`` method (KLRClassifier), the method
    is called with the validation split when ``cross_validate_KLR=True``.
    All other classifiers are fitted with the standard sklearn ``fit`` API.

    Parameters
    ----------
    data             : sklearn Bunch with .data and .target attributes
    classifiers      : dict mapping name -> classifier instance
    test_size        : fraction of data held out for testing
    cross_validate_KLR : if True, reserve half the test split for KLR tuning
    random_state     : random seed
    verbose          : print per-classifier progress
    noise_scale      : std of Gaussian noise added to X (0 = no noise)
    """
    X, y = data.data, data.target
    y = LabelEncoder().fit_transform(y)

    if noise_scale > 0:
        rng = np.random.RandomState(random_state)
        X = X + rng.normal(0, noise_scale, size=X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if cross_validate_KLR:
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=random_state
        )
    else:
        X_val = y_val = None

    print(f"Train size:      {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test size:       {X_test.shape[0]} samples")
    if X_val is not None:
        print(f"Validation size: {X_val.shape[0]} samples")

    results = []
    for name, clf in classifiers.items():
        if verbose:
            tqdm.write(f"Training {name}...")

        t0 = time.time()

        if hasattr(clf, "tune") and cross_validate_KLR and X_val is not None:
            clf = clf.tune(X_train, y_train, X_val, y_val, verbose=verbose)
        else:
            clf = clf.fit(X_train, y_train)

        runtime       = time.time() - t0
        train_accuracy = clf.score(X_train, y_train)
        test_accuracy  = clf.score(X_test,  y_test)

        if verbose:
            tqdm.write("  done.")

        results.append({
            "Classifier":      name,
            "Train Accuracy":  round(train_accuracy, 3),
            "Test Accuracy":   round(test_accuracy,  3),
            "Runtime (s)":     round(runtime, 3),
        })

    table = PrettyTable(["Classifier", "Train Accuracy", "Test Accuracy", "Runtime (s)"])
    for r in results:
        table.add_row([r["Classifier"], r["Train Accuracy"], r["Test Accuracy"], r["Runtime (s)"]])
    print(table)


def plot_decision_boundaries(
    datasets: list,
    classifiers: dict,
    random_state: int = 42,
    KLR_crossvalidation: bool = False,
    test_size: float = 0.25,
    resolution: int = 75,
):
    """
    Plot decision boundaries for each (dataset, classifier) pair.

    The first column always shows the raw data scatter.  Decision regions
    are drawn with per-class colourmaps; test accuracy (large) and train
    accuracy (small, faded) are annotated in each subplot.

    For classifiers with a ``tune`` method and when ``KLR_crossvalidation``
    is True, half the test split is used as a validation set for tuning.

    Parameters
    ----------
    datasets             : list of (X, y) tuples
    classifiers          : dict mapping name -> classifier instance
    random_state         : random seed
    KLR_crossvalidation  : enable tune() for eligible classifiers
    test_size            : fraction for test (and validation) split
    resolution           : grid resolution for decision-boundary contours
    """
    n_rows = len(datasets)
    n_cols = len(classifiers) + 1
    _, axes = plt.subplots(figsize=(4 * n_cols, 4 * n_rows), ncols=n_cols, nrows=n_rows)

    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, (X, y) in enumerate(datasets):
        n_unique = len(np.unique(y))
        if n_unique == 2:
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            cm = [plt.cm.Reds, plt.cm.Blues]
        else:
            cm_bright = ListedColormap(mpl_colormaps["tab10"].colors[:n_unique])
            cm = [
                plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Reds,
                plt.cm.Purples, plt.cm.Greys, plt.cm.YlOrBr, plt.cm.YlGnBu,
                plt.cm.PuRd, plt.cm.BuGn,
            ]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        if KLR_crossvalidation:
            X_val, X_test, y_val, y_test = train_test_split(
                X_test, y_test, test_size=0.5, random_state=random_state
            )
        else:
            X_val = y_val = None

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        for j, (name, clf) in enumerate(classifiers.items(), start=1):
            if hasattr(clf, "tune") and KLR_crossvalidation and X_val is not None:
                clf = clf.tune(X_train, y_train, X_val, y_val, verbose=False)
            else:
                clf = clf.fit(X_train, y_train)

            test_score  = clf.score(X_test,  y_test)
            train_score = clf.score(X_train, y_train)

            # predict_proba returns (n_grid, n_classes) with columns in
            # clf.classes_ order; for integer labels starting at 0, column
            # index equals class label, so probs[:, lab] is correct.
            probs  = clf.predict_proba(grid)
            y_pred = clf.predict(grid).reshape(xx.shape)

            for colormap, lab in zip(cm, np.sort(np.unique(y_pred))):
                # Find column index for this label
                col_idx = np.searchsorted(clf.classes_, lab)
                labprob = probs[:, col_idx].reshape(xx.shape)
                axes[i, j].contourf(
                    xx, yy,
                    np.where(y_pred == lab, labprob, np.nan),
                    cmap=colormap,
                    alpha=0.8,
                )

            axes[i, j].set_title(name)
            axes[i, j].text(
                x_max - 0.05 * (x_max - x_min),
                y_min + 0.15 * (y_max - y_min),
                f"{test_score:.2f}".lstrip("0"),
                size=15, horizontalalignment="right",
            )
            axes[i, j].text(
                x_max - 0.05 * (x_max - x_min),
                y_min + 0.05 * (y_max - y_min),
                f"{train_score:.2f}".lstrip("0"),
                size=12, horizontalalignment="right", alpha=0.7,
            )

        # Scatter and axis formatting for every column
        for j in range(n_cols):
            ax = axes[i, j]
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
            ax.scatter(X_test[:,  0], X_test[:,  1], c=y_test,  cmap=cm_bright, alpha=0.6, edgecolors="k")

        axes[i, 0].set_title("Input data")

    plt.tight_layout()
    plt.show()
