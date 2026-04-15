"""
Kernel Likelihood Ratio Classifiers (KLRC)
==========================================
Implements the maximum-likelihood classifier in the RKHS derived from the
mutual-singularity / separation-of-measure phenomenon (Santoro, Waghmare,
Panaretos 2025, 2026).

Given class-conditional distributions {P_j}, each embedded as a Gaussian
measure N(m_Pj, C_Pj) on the RKHS H, the classifier assigns a test point x
to the class with the highest regularised log-likelihood:

    ĵ(x) = argmin_j { (k_x - m_j)^T (C_j + γI)^{-1} (k_x - m_j)
                      + log|C_j + γI| }

where k_x = [k(x, x_1), …, k(x, x_n)]^T is the empirical feature vector of x,
and m_j, C_j are the empirical mean and covariance of class j in that
representation.  The ridge γ > 0 ensures well-posedness and governs the
overfitting / stability trade-off (see KLRClassifier docstring).

Classes
-------
KLRClassifier        : base classifier (sklearn-compatible API).
BaggingKLRClassifier : ensemble with implicit spectral pruning.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy import linalg as sp_linalg
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

__all__ = ["KLRClassifier", "BaggingKLRClassifier", "NystromKLRClassifier"]


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean pairwise distances D[i, j] = ||X[i] - Y[j]||^2.

    Uses torch on GPU when available for speed; always returns a float64
    numpy array of shape (len(X), len(Y)).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    D = torch.cdist(Xt, Yt, p=2).pow(2)
    return D.cpu().numpy().astype(np.float64)


def _rbf(D: np.ndarray, bandwidth: float) -> np.ndarray:
    """K[i, j] = exp(-D[i, j] / bandwidth)."""
    return np.exp(-D / bandwidth)


def _chol_logdet(L: np.ndarray) -> float:
    """log|A| = 2 * sum(log diag(L)) for A = L L^T (L lower triangular)."""
    return 2.0 * float(np.log(np.diag(L)).sum())


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax. Input shape: (n, k)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# KLRClassifier
# ---------------------------------------------------------------------------

class KLRClassifier(BaseEstimator, ClassifierMixin):
    """
    Kernel Likelihood Ratio Classifier.

    Assigns each observation to the class j whose regularised Gaussian
    embedding in the RKHS yields the highest log-likelihood:

        ĵ(x) = argmin_j { (k_x - m_j)^T (C_j + γI)^{-1} (k_x - m_j)
                          + log|C_j + γI| }

    The empirical feature vector k_x = [k(x, x_1), …, k(x, x_n)]^T lives in
    R^n (the span of the training kernel evaluations).  m_j and C_j are the
    empirical mean and covariance of the n_j × n kernel sub-matrix for class j.

    Regularisation trade-off
    ------------------------
    The ridge γ directly controls the discrimination sharpness vs. stability:

    γ → 0 :
        Approaches the theoretical "perfect" classifier guaranteed by the
        mutual-singularity result.  The inverse (C_j + γI)^{-1} amplifies all
        directions in the RKHS, including those with near-zero variance that
        are purely noise, making the boundary very sharp but numerically
        ill-conditioned and prone to overfitting the empirical support.

    γ → ∞ :
        C_j + γI ≈ γI for every class, so the log-det terms cancel and the
        rule collapses to kernel nearest-centroid (ignoring all second-order
        information).  Numerically stable but loses the RKHS covariance
        structure that drives the separation-of-measure phenomenon.

    Optimal γ lies between these extremes and can be located efficiently
    with the ``tune`` method, which caches distances to avoid redundant
    recomputation across the grid.

    When ``ridge_mode='spectral'``, γ is interpreted as a fraction of the
    mean largest eigenvalue of the class covariances, making the
    regularisation scale-invariant across datasets and kernel bandwidths.
    A value of ridge=0.01 then means "1 % of the dominant spectral scale",
    giving a principled default that transfers across problems.

    Parameters
    ----------
    ridge : float, default=1e-3
        Tikhonov regularisation strength γ.
    band_factor : float, default=1.0
        Bandwidth multiplier h = 2 * median_pairwise_dist * band_factor.
        Large values smooth the kernel (underfitting); small values sharpen it.
    center_covariance : bool, default=True
        True  → centered empirical covariance  C = Cov(K_j)  (standard).
        False → second moment  C = K_j^T K_j / n_j  (uncentered; matches the
                theoretical C_P = E[φ(X) ⊗ φ(X)] when the mean embedding is
                separately accounted for).
    ridge_mode : {'absolute', 'spectral'}, default='absolute'
        'absolute' : use ridge as-is (units: kernel-matrix scale).
        'spectral'  : multiply ridge by the mean largest eigenvalue of the
                      class covariances before adding to each C_j.
    store_covariance : bool, default=False
        If True, retain ``self.covs_`` (list of n×n arrays) after fitting.
        If False (default), the covariance matrices are discarded once the
        Cholesky factors have been computed, halving the persistent memory
        from  2·C·n²·8  to  C·n²·8  bytes.  Only set to True if you need
        to inspect the raw covariance matrices post-fit.

    Attributes (set after fit)
    --------------------------
    classes_         : ndarray of shape (n_classes,)
    means_           : list of ndarray, each shape (n_train,)
    covs_            : list of ndarray  (only present when store_covariance=True)
    chols_           : list of Cholesky factors (L, lower) of C_j + γ_eff I
    log_dets_        : list of float, log|C_j + γ_eff I| per class
    effective_ridge_ : float, γ actually used (equals ridge in absolute mode)
    bandwidth_       : float
    median_dist_     : float, median pairwise squared distance on training set
    X_train_         : ndarray
    """

    def __init__(
        self,
        ridge: float = 1e-3,
        band_factor: float = 1.0,
        center_covariance: bool = True,
        ridge_mode: str = "absolute",
        store_covariance: bool = False,
    ) -> None:
        self.ridge = ridge
        self.band_factor = band_factor
        self.center_covariance = center_covariance
        self.ridge_mode = ridge_mode
        self.store_covariance = store_covariance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _class_stats(
        self, K: np.ndarray, y_enc: np.ndarray, n_classes: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute per-class means and covariances from kernel matrix K."""
        means, covs = [], []
        for c in range(n_classes):
            Kc = K[y_enc == c, :]  # n_c × n
            means.append(Kc.mean(axis=0))
            if self.center_covariance:
                covs.append(np.cov(Kc, rowvar=False, bias=True))
            else:
                covs.append(Kc.T @ Kc / float(Kc.shape[0]))
        return means, covs

    def _effective_ridge(self, covs: List[np.ndarray]) -> float:
        """Return γ_eff = ridge * spectral_scale (or ridge if absolute)."""
        if self.ridge_mode == "spectral":
            n = covs[0].shape[0]
            max_eigs = [
                float(
                    sp_linalg.eigvalsh(
                        C, subset_by_index=[n - 1, n - 1], check_finite=False
                    )
                )
                for C in covs
            ]
            scale = float(np.mean(max_eigs))
            if scale <= 0.0:
                warnings.warn(
                    "Non-positive spectral scale; reverting to absolute ridge.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return float(self.ridge)
            return float(self.ridge) * scale
        return float(self.ridge)

    @staticmethod
    def _factorize(
        C: np.ndarray, eff_ridge: float
    ) -> Tuple[Tuple[np.ndarray, bool], float]:
        """
        Cholesky-factorize C + eff_ridge * I.

        Returns
        -------
        factor   : (L, lower) suitable for scipy.linalg.cho_solve
        log_det  : log|C + eff_ridge * I|
        """
        n = C.shape[0]
        RC = C + eff_ridge * np.eye(n)
        try:
            factor = sp_linalg.cho_factor(RC, lower=True, check_finite=False)
        except sp_linalg.LinAlgError as exc:
            raise sp_linalg.LinAlgError(
                f"Cholesky factorisation failed for class covariance "
                f"(effective ridge = {eff_ridge:.2e}).  "
                "Try increasing `ridge` or switching to ridge_mode='spectral'."
            ) from exc
        log_det = _chol_logdet(factor[0])
        return factor, log_det

    def _score_test(
        self,
        kx: np.ndarray,
    ) -> np.ndarray:
        """
        Compute decision scores given kernel matrix kx (n_train × n_test).

        Returns scores of shape (n_classes, n_test).
        """
        n_classes = len(self.classes_)
        n_test = kx.shape[1]
        scores = np.empty((n_classes, n_test))
        for j, (m, factor, logdet) in enumerate(
            zip(self.means_, self.chols_, self.log_dets_)
        ):
            diff = kx - m[:, np.newaxis]               # n_train × n_test
            sol = sp_linalg.cho_solve(factor, diff, check_finite=False)
            scores[j] = np.einsum("ij,ij->j", diff, sol) + logdet
        return scores

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KLRClassifier":
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        y_enc = self.label_encoder_.transform(y)
        n_classes = len(self.classes_)

        self.X_train_ = X
        D = _sq_dists(X, X)
        pos = D[D > 0.0]
        if pos.size == 0:
            raise ValueError(
                "All pairwise distances are zero; check that X is not constant."
            )
        self.median_dist_ = float(np.median(pos))
        self.bandwidth_ = 2.0 * self.median_dist_ * float(self.band_factor)

        K = _rbf(D, self.bandwidth_)                            # n × n
        del D                                                   # free n×n temp
        means, covs = self._class_stats(K, y_enc, n_classes)
        del K                                                   # free n×n temp
        self.means_ = means

        self.effective_ridge_ = self._effective_ridge(covs)

        self.chols_, self.log_dets_ = [], []
        for C in covs:
            factor, logdet = self._factorize(C, self.effective_ridge_)
            self.chols_.append(factor)
            self.log_dets_.append(logdet)

        # Retain raw covariances only when explicitly requested.
        # Default (False) halves persistent memory: C·n²·8 bytes instead of 2·C·n²·8.
        if self.store_covariance:
            self.covs_ = covs
        else:
            del covs

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Regularised log-likelihood scores for each class.

        score[j, i] is the negative log-likelihood of sample i under the
        class-j Gaussian embedding.  Lower score = higher likelihood.

        Parameters
        ----------
        X : array of shape (n_test, n_features)

        Returns
        -------
        scores : array of shape (n_classes, n_test)
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        D = _sq_dists(self.X_train_, X)          # n_train × n_test
        kx = _rbf(D, self.bandwidth_)
        return self._score_test(kx)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels. Returns array of shape (n_test,)."""
        check_is_fitted(self)
        return self.label_encoder_.inverse_transform(
            np.argmin(self.decision_function(X), axis=0)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class-membership probabilities.

        Computed as softmax of the negative decision scores, so that higher
        likelihood → higher probability.  Returns array of shape
        (n_test, n_classes); columns follow the order of ``self.classes_``.
        """
        check_is_fitted(self)
        logits = -self.decision_function(X).T    # (n_test, n_classes)
        return _softmax_rows(logits)

    def score(self, X: np.ndarray, y) -> float:
        """Mean accuracy on (X, y)."""
        check_is_fitted(self)
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ridge_grid: Optional[List[float]] = None,
        band_factor_grid: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> "KLRClassifier":
        """
        Fast grid search over (ridge, band_factor) on a held-out validation set.

        Design
        ------
        Pairwise squared distances are computed *once* and reused across all
        grid combinations.  For each candidate bandwidth, the kernel matrix is
        built and class statistics recomputed; for each candidate ridge, only
        the Cholesky factorisations and validation scores are refreshed.  This
        avoids the dominant O(n²) distance computation at every grid point.

        The search is organised as::

            for band_factor in band_factor_grid:          ← recompute K, means, covs
                [if spectral: compute spectral scale once]
                for ridge in ridge_grid:                  ← recompute chols, scores
                    evaluate on X_val

        After the search, ``self.ridge`` and ``self.band_factor`` are updated
        and the model is re-fitted on X_train.

        Regularisation interpretation
        ------------------------------
        When ``ridge_mode='absolute'``, the grid sweeps raw γ values.
        When ``ridge_mode='spectral'``, γ values are multiplied by the mean
        largest eigenvalue of the class covariances (computed at each bandwidth),
        so the grid sweeps scale-invariant fractions of the dominant spectral
        energy (e.g. ridge=0.01 → penalise at 1 % of the covariance scale).

        Parameters
        ----------
        X_train, y_train : training arrays
        X_val,   y_val   : held-out validation arrays
        ridge_grid       : values of ridge to search
                           (default: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        band_factor_grid : values of band_factor to search
                           (default: [100, 50, 10, 5, 2, 1, 0.5, 0.1, 0.05])
        verbose          : print selected parameters and validation accuracy

        Returns
        -------
        self, fitted on X_train with the best (ridge, band_factor)
        """
        if ridge_grid is None:
            ridge_grid = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        if band_factor_grid is None:
            band_factor_grid = [100, 50, 10, 5, 2, 1, 0.5, 0.1, 0.05]

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)
        X_val   = np.asarray(X_val,   dtype=float)
        y_val   = np.asarray(y_val)

        le      = LabelEncoder().fit(y_train)
        y_enc   = le.transform(y_train)
        classes = le.classes_
        n_classes = len(classes)
        n       = X_train.shape[0]
        n_val   = X_val.shape[0]
        I_n     = np.eye(n)

        # --- compute distances once ----------------------------------------
        D_train = _sq_dists(X_train, X_train)
        D_val   = _sq_dists(X_train, X_val)    # n_train × n_val
        pos     = D_train[D_train > 0.0]
        median_dist = float(np.median(pos))

        best_acc   = -1.0
        best_ridge = float(self.ridge)
        best_bf    = float(self.band_factor)

        for bf in band_factor_grid:
            bandwidth = 2.0 * median_dist * float(bf)
            K_train   = _rbf(D_train, bandwidth)
            K_val     = _rbf(D_val,   bandwidth)

            # Class statistics for this bandwidth
            means, covs = [], []
            for c in range(n_classes):
                mask = y_enc == c
                Kc   = K_train[mask, :]
                means.append(Kc.mean(axis=0))
                if self.center_covariance:
                    covs.append(np.cov(Kc, rowvar=False, bias=True))
                else:
                    covs.append(Kc.T @ Kc / float(Kc.shape[0]))

            # Spectral scale (once per bandwidth)
            if self.ridge_mode == "spectral":
                max_eigs = [
                    float(
                        sp_linalg.eigvalsh(
                            C, subset_by_index=[n - 1, n - 1], check_finite=False
                        )
                    )
                    for C in covs
                ]
                spectral_scale = max(float(np.mean(max_eigs)), 1e-12)
            else:
                spectral_scale = 1.0

            for ridge in ridge_grid:
                eff_ridge = float(ridge) * spectral_scale
                scores    = np.full((n_classes, n_val), np.inf)
                ok        = True

                for j, (m, C) in enumerate(zip(means, covs)):
                    try:
                        factor = sp_linalg.cho_factor(
                            C + eff_ridge * I_n, lower=True, check_finite=False
                        )
                    except sp_linalg.LinAlgError:
                        ok = False
                        break
                    logdet = _chol_logdet(factor[0])
                    diff   = K_val - m[:, np.newaxis]
                    sol    = sp_linalg.cho_solve(factor, diff, check_finite=False)
                    scores[j] = np.einsum("ij,ij->j", diff, sol) + logdet

                if not ok:
                    continue

                y_pred = le.inverse_transform(np.argmin(scores, axis=0))
                acc    = float(np.mean(y_pred == y_val))

                if acc > best_acc:
                    best_acc   = acc
                    best_ridge = float(ridge)
                    best_bf    = float(bf)

        if verbose:
            print(
                f"[KLRClassifier.tune]  "
                f"ridge={best_ridge:.2e},  band_factor={best_bf},  "
                f"val_acc={best_acc:.4f}"
            )

        self.set_params(ridge=best_ridge, band_factor=best_bf)
        return self.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# BaggingKLRClassifier
# ---------------------------------------------------------------------------

class BaggingKLRClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of KLRClassifiers trained on random subsamples.

    Aggregates decision scores across bags by averaging, then takes argmin:

        score_agg(x, j) = (1/B) * Σ_b score_b(x, j)

    Spectral-pruning effect
    -----------------------
    Each base classifier is trained on m << n samples, so its empirical
    covariance C_j^{(b)} has rank at most min(m, n_c^{(b)}).  The
    high-variance (noise) tail of the spectrum is discarded implicitly.
    Averaging B such classifiers reduces the estimator variance without
    explicit rank truncation, analogous to Nyström-type approximations.

    The bag_size parameter controls the pruning intensity:
    - Smaller bag_size → lower rank per bag → stronger spectral pruning →
      lower per-bag variance, higher bias.
    - Larger bag_size → higher rank per bag → weaker pruning → lower bias,
      higher per-bag variance (though averaging over B bags mitigates this).

    Optional column subsampling via ``feature_fraction`` provides an
    orthogonal, feature-space pruning axis: reducing the effective input
    dimension decreases the kernel matrix size O(m² → (m·p')²) and therefore
    the covariance rank.

    Parameters
    ----------
    base_estimator : KLRClassifier or None, default=None
        Prototype cloned for each bag.  Defaults to KLRClassifier() (using
        its own ridge / band_factor defaults).
    n_estimators : int, default=10
        Number of bags B.
    bag_size : int or float, default=0.5
        Samples per bag.
        - int   : absolute count.
        - float in (0, 1] : fraction of n_train.
        Sampling is stratified so that every class appears in every bag.
    feature_fraction : float or None, default=None
        Fraction of features sampled per bag (column subsampling).
        None = use all features.
    n_jobs : int, default=1
        Parallel jobs for bag fitting via joblib (-1 = all CPUs).
    random_state : int or None, default=None

    Attributes (set after fit)
    --------------------------
    estimators_      : list of KLRClassifier, length n_estimators
    feature_indices_ : list of ndarray, feature subset per bag
    classes_         : ndarray of all class labels
    """

    def __init__(
        self,
        base_estimator: Optional[KLRClassifier] = None,
        n_estimators: int = 10,
        bag_size: Union[int, float] = 0.5,
        feature_fraction: Optional[float] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.base_estimator    = base_estimator
        self.n_estimators      = n_estimators
        self.bag_size          = bag_size
        self.feature_fraction  = feature_fraction
        self.n_jobs            = n_jobs
        self.random_state      = random_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_base(self) -> KLRClassifier:
        return (
            self.base_estimator
            if self.base_estimator is not None
            else KLRClassifier()
        )

    def _bag_size_int(self, n: int, n_classes: int) -> int:
        m = (
            int(round(self.bag_size * n))
            if isinstance(self.bag_size, float)
            else int(self.bag_size)
        )
        # Must be large enough to contain at least one sample per class
        return int(np.clip(m, n_classes, n))

    def _fit_bag(
        self, X: np.ndarray, y: np.ndarray, seed: int
    ) -> Tuple["KLRClassifier", np.ndarray]:
        """Fit one base estimator on a stratified subsample."""
        rng      = np.random.RandomState(seed)
        n        = X.shape[0]
        classes  = np.unique(y)
        m        = self._bag_size_int(n, len(classes))

        # Stratified subsample without replacement
        if m < n:
            try:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=m,
                    random_state=int(rng.randint(0, 2**31 - 1)),
                )
                idx, _ = next(sss.split(np.zeros(n), y))
            except ValueError:
                # Fallback when stratification fails (e.g. single-sample class)
                idx = rng.choice(n, size=m, replace=False)
        else:
            idx = np.arange(n)

        X_bag, y_bag = X[idx], y[idx]

        # Optional feature subsampling
        if self.feature_fraction is not None:
            p        = X.shape[1]
            k        = max(1, int(round(self.feature_fraction * p)))
            feat_idx = np.sort(rng.choice(p, size=k, replace=False))
        else:
            feat_idx = np.arange(X.shape[1])

        est = clone(self._resolve_base())
        est.fit(X_bag[:, feat_idx], y_bag)
        return est, feat_idx

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingKLRClassifier":
        """
        Fit all bags in parallel.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_        = self.label_encoder_.classes_

        rng   = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**31 - 1, size=self.n_estimators)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_bag)(X, y, int(s)) for s in seeds
        )
        self.estimators_, self.feature_indices_ = zip(*results)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Average decision scores across all bags.

        Each bag contributes scores only for the classes it observed.
        Scores are accumulated per-class and divided by the number of bags
        that saw each class; classes absent from every bag receive score +∞.

        Returns
        -------
        scores : array of shape (n_classes, n_test)
        """
        check_is_fitted(self)
        X         = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        n_test    = X.shape[0]

        agg   = np.zeros((n_classes, n_test))
        count = np.zeros(n_classes, dtype=int)

        for est, feat_idx in zip(self.estimators_, self.feature_indices_):
            bag_scores  = est.decision_function(X[:, feat_idx])  # (k, n_test)
            global_idx  = np.searchsorted(self.classes_, est.classes_)
            agg[global_idx]   += bag_scores
            count[global_idx] += 1

        # Normalise; mark classes never seen by any bag as +inf
        for j in range(n_classes):
            if count[j] > 0:
                agg[j] /= count[j]
            else:
                agg[j] = np.inf
                warnings.warn(
                    f"Class '{self.classes_[j]}' was absent from all bags. "
                    "Increase bag_size or n_estimators.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return agg

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels. Returns array of shape (n_test,)."""
        check_is_fitted(self)
        return self.label_encoder_.inverse_transform(
            np.argmin(self.decision_function(X), axis=0)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class-membership probabilities (softmax of negative averaged scores).
        Returns array of shape (n_test, n_classes).
        """
        check_is_fitted(self)
        logits = -self.decision_function(X).T    # (n_test, n_classes)
        return _softmax_rows(logits)

    def score(self, X: np.ndarray, y) -> float:
        """Mean accuracy on (X, y)."""
        check_is_fitted(self)
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ---------------------------------------------------------------------------
# Nyström utilities  (shared by NystromKLRClassifier and its tune method)
# ---------------------------------------------------------------------------

def _nystrom_decompose(
    K_mm: np.ndarray, reg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regularised eigendecomposition of K_mm.

    Returns
    -------
    vecs          : ndarray (m, rank)  — eigenvectors
    inv_sqrt_vals : ndarray (rank,)    — 1/sqrt(eigenvalue)
    """
    m = K_mm.shape[0]
    vals, vecs = sp_linalg.eigh(K_mm + reg * np.eye(m), check_finite=False)
    keep = vals > 100.0 * reg          # discard near-zero components
    return vecs[:, keep], 1.0 / np.sqrt(vals[keep])


def _nystrom_phi(
    K_nm: np.ndarray,
    vecs: np.ndarray,
    isv: np.ndarray,
) -> np.ndarray:
    """Nyström feature matrix Phi = K_nm @ vecs * isv.  Shape: (n, rank)."""
    return (K_nm @ vecs) * isv[np.newaxis, :]


# ---------------------------------------------------------------------------
# NystromKLRClassifier
# ---------------------------------------------------------------------------

class NystromKLRClassifier(KLRClassifier):
    """
    KLR Classifier with Nyström kernel approximation.

    Replaces the n×n full kernel matrix with an n×m Nyström feature matrix
    (m << n landmark points), reducing persistent memory from

        C · n² · 8  bytes   (full KLR Cholesky factors)

    to

        C · m² · 8  bytes   (Nyström Cholesky factors)
      + m · d · 8  bytes   (landmarks, needed for test-time kernel)

    For n = 50 000, m = 500, C = 10:  ~20 MB vs ~20 GB.

    Approximation quality increases with m; at m = n the decomposition
    is exact.  In practice m ~ 500-2000 recovers most of the full-KLR
    accuracy with a fraction of the cost.

    The ``tune`` method caches landmark-distance matrices (O(nm) once)
    so the grid search costs O(m²·|grid|) instead of O(n²·|grid|).

    Parameters
    ----------
    ridge, band_factor, center_covariance, ridge_mode, store_covariance
        Inherited from KLRClassifier.
    n_landmarks : int, default=500
    nystrom_reg : float, default=1e-8
        Ridge on K_mm for stability; components below 100·nystrom_reg dropped.
    random_state : int or None, default=None
    """

    def __init__(
        self,
        ridge: float = 1e-3,
        band_factor: float = 1.0,
        center_covariance: bool = True,
        ridge_mode: str = "absolute",
        store_covariance: bool = False,
        n_landmarks: int = 500,
        nystrom_reg: float = 1e-8,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            ridge=ridge, band_factor=band_factor,
            center_covariance=center_covariance,
            ridge_mode=ridge_mode, store_covariance=store_covariance,
        )
        self.n_landmarks  = n_landmarks
        self.nystrom_reg  = nystrom_reg
        self.random_state = random_state

    # ---- helpers ----------------------------------------------------------

    def _select_landmarks(self, n: int, y_enc: np.ndarray, n_classes: int) -> np.ndarray:
        """Stratified random landmark indices (no replacement)."""
        m   = min(self.n_landmarks, n)
        rng = np.random.RandomState(self.random_state)
        idx = []
        for c in range(n_classes):
            ci   = np.where(y_enc == c)[0]
            take = min(max(1, int(round(m * len(ci) / n))), len(ci))
            idx.extend(rng.choice(ci, size=take, replace=False).tolist())
        idx = list(set(idx))
        if len(idx) < m:
            rest  = np.setdiff1d(np.arange(n), idx)
            extra = rng.choice(rest, size=min(m - len(idx), len(rest)), replace=False)
            idx.extend(extra.tolist())
        return np.array(idx[:m])

    def _median_sub(self, X: np.ndarray, n_sub: int = 2000) -> float:
        """Median pairwise sq-distance from a random subsample (avoids O(n²))."""
        n = X.shape[0]
        if n > n_sub:
            X = X[np.random.RandomState(self.random_state).choice(n, n_sub, replace=False)]
        D   = _sq_dists(X, X)
        pos = D[D > 0.0]
        return float(np.median(pos))

    def _fit_phi(self, Phi: np.ndarray, y_enc: np.ndarray, n_classes: int) -> None:
        """Fit class statistics from (n × rank) feature matrix."""
        means, covs = [], []
        for c in range(n_classes):
            Phi_c = Phi[y_enc == c, :]
            means.append(Phi_c.mean(axis=0))
            if self.center_covariance:
                covs.append(np.cov(Phi_c, rowvar=False, bias=True))
            else:
                covs.append(Phi_c.T @ Phi_c / float(Phi_c.shape[0]))
        self.means_           = means
        self.effective_ridge_ = self._effective_ridge(covs)
        self.chols_, self.log_dets_ = [], []
        for C in covs:
            factor, logdet = self._factorize(C, self.effective_ridge_)
            self.chols_.append(factor)
            self.log_dets_.append(logdet)
        if self.store_covariance:
            self.covs_ = covs

    # ---- sklearn API ------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NystromKLRClassifier":
        """
        Fit with Nyström approximation.

        Peak memory during fit: O(nm) for K_nm (freed after class stats).
        Persistent memory:      O(Cm²) Cholesky + O(md) landmarks.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_        = self.label_encoder_.classes_
        y_enc                = self.label_encoder_.transform(y)
        n_classes            = len(self.classes_)

        self.median_dist_ = self._median_sub(X)
        self.bandwidth_   = 2.0 * self.median_dist_ * float(self.band_factor)

        lm_idx          = self._select_landmarks(X.shape[0], y_enc, n_classes)
        self.landmarks_ = X[lm_idx]

        D_mm = _sq_dists(self.landmarks_, self.landmarks_)
        self.nystrom_vecs_, self.nystrom_inv_sqrt_vals_ = _nystrom_decompose(
            _rbf(D_mm, self.bandwidth_), self.nystrom_reg
        )
        self.nystrom_rank_ = len(self.nystrom_inv_sqrt_vals_)
        del D_mm

        D_nm = _sq_dists(X, self.landmarks_)
        Phi  = _nystrom_phi(_rbf(D_nm, self.bandwidth_),
                            self.nystrom_vecs_, self.nystrom_inv_sqrt_vals_)
        del D_nm

        self._fit_phi(Phi, y_enc, n_classes)
        del Phi
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Shape: (n_classes, n_test).  Test cost: O(m·n_test) per class."""
        check_is_fitted(self)
        X   = np.asarray(X, dtype=float)
        Phi = _nystrom_phi(
            _rbf(_sq_dists(X, self.landmarks_), self.bandwidth_),
            self.nystrom_vecs_, self.nystrom_inv_sqrt_vals_,
        )
        return self._score_test(Phi.T)   # (rank × n_test)

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ridge_grid: Optional[List[float]] = None,
        band_factor_grid: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> "NystromKLRClassifier":
        """
        Grid search over (ridge, band_factor) with cached landmark distances.

        Cost: O(nm) once for distances, then O(m³C·|grid|) for scoring.
        Versus O(n²) + O(n³C·|grid|) for full KLR.tune.
        """
        if ridge_grid        is None: ridge_grid        = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        if band_factor_grid  is None: band_factor_grid  = [100, 50, 10, 5, 2, 1, 0.5, 0.1, 0.05]

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)
        X_val   = np.asarray(X_val,   dtype=float)
        y_val   = np.asarray(y_val)

        le        = LabelEncoder().fit(y_train)
        y_enc     = le.transform(y_train)
        n_classes = len(le.classes_)
        n_val     = X_val.shape[0]

        lm_idx    = self._select_landmarks(X_train.shape[0], y_enc, n_classes)
        lm        = X_train[lm_idx]
        median    = self._median_sub(X_train)

        D_mm    = _sq_dists(lm,      lm)
        D_nm    = _sq_dists(X_train, lm)
        D_val_m = _sq_dists(X_val,   lm)

        best_acc, best_ridge, best_bf = -1.0, float(self.ridge), float(self.band_factor)

        for bf in band_factor_grid:
            bw    = 2.0 * median * float(bf)
            vecs, isv = _nystrom_decompose(_rbf(D_mm, bw), self.nystrom_reg)
            rank  = len(isv)
            I_r   = np.eye(rank)

            Phi_tr  = _nystrom_phi(_rbf(D_nm,    bw), vecs, isv)
            Phi_val = _nystrom_phi(_rbf(D_val_m, bw), vecs, isv)

            means, covs = [], []
            for c in range(n_classes):
                Phi_c = Phi_tr[y_enc == c, :]
                means.append(Phi_c.mean(axis=0))
                if self.center_covariance:
                    covs.append(np.cov(Phi_c, rowvar=False, bias=True))
                else:
                    covs.append(Phi_c.T @ Phi_c / float(Phi_c.shape[0]))

            if self.ridge_mode == "spectral":
                spectral_scale = max(float(np.mean([
                    float(sp_linalg.eigvalsh(C, subset_by_index=[rank-1, rank-1],
                                             check_finite=False))
                    for C in covs
                ])), 1e-12)
            else:
                spectral_scale = 1.0

            Pv = Phi_val.T
            for ridge in ridge_grid:
                er     = float(ridge) * spectral_scale
                scores = np.full((n_classes, n_val), np.inf)
                ok     = True
                for j, (m_j, C) in enumerate(zip(means, covs)):
                    try:
                        fac = sp_linalg.cho_factor(C + er * I_r, lower=True, check_finite=False)
                    except sp_linalg.LinAlgError:
                        ok = False; break
                    diff      = Pv - m_j[:, np.newaxis]
                    sol       = sp_linalg.cho_solve(fac, diff, check_finite=False)
                    scores[j] = np.einsum("ij,ij->j", diff, sol) + _chol_logdet(fac[0])
                if not ok: continue
                acc = float(np.mean(le.inverse_transform(np.argmin(scores, axis=0)) == y_val))
                if acc > best_acc:
                    best_acc, best_ridge, best_bf = acc, float(ridge), float(bf)

        if verbose:
            print(f"[NystromKLRClassifier.tune]  "
                  f"ridge={best_ridge:.2e},  band_factor={best_bf},  val_acc={best_acc:.4f}")

        self.set_params(ridge=best_ridge, band_factor=best_bf)
        return self.fit(X_train, y_train)
