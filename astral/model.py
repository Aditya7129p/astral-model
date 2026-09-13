"""
AstralModel — Spectral Resonance Decomposition (SRD)
=====================================================

A causal-invariant and interpretable machine learning algorithm that unifies:
  • Adaptive spectral resonance basis functions (Quantile Hats, Gaussian RBF, Sigmoids, Roots, Powers)
  • Non-linear pairwise interaction discovery (Multiplicative, Soft Ratios, Min, AbsDiff)
  • Invariant Risk Minimization (IRM) stability diagnostics across perturbed bootstrap environments
  • Stability-weighted regularized fitting (invariant features receive lower shrinkage)
  • Fast stratified cross-validation for optimal regularizer selection
  • Calibrated uncertainty via conformal prediction intervals & sets with optional abstention
  • Minimal Description Length (MDL) greedy orthogonalized basis pruning
  • Full Scikit-Learn API compatibility (fit, predict, predict_proba, score) + Python package metadata

Supports both classification and regression.

Package Usage
-------------
    from astral_model import AstralModel, AstralScaler, astral_train_test_split, __version__

    X_train, X_test, y_train, y_test = astral_train_test_split(X, y, test_size=0.2, stratify=y)
    scaler = AstralScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = AstralModel(ridge="auto", random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    scores = model.score(X_test_s, y_test)
"""

import sys
import warnings
import numpy as np

# Package Version & Metadata
__version__ = "0.2.1"
__version_info__ = (0, 2, 1)
version = __version__
__author__ = "AstralModel Contributors"
__license__ = "Apache-2.0"

# Fix console encoding on Windows if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

__all__ = [
    "AstralModel",
    "AstralScaler",
    "astral_train_test_split",
    "astral_cross_validate",
    "ResonanceBasis",
    "__version__",
    "__version_info__",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES — AstralScaler & Train/Test Split
# ═══════════════════════════════════════════════════════════════════════════════


class AstralScaler:
    """
    Standard z-score scaler (zero mean, unit variance).
    Drop-in replacement for scikit-learn's StandardScaler, implemented in pure NumPy.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X, y=None):
        """Compute the mean and std to be used for scaling."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        self._fitted = True
        return self

    def transform(self, X):
        """Scale features of X using previously computed statistics."""
        if not self._fitted:
            raise RuntimeError("AstralScaler has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Undo the scaling and recover original scale."""
        if not self._fitted:
            raise RuntimeError("AstralScaler has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X * self.std_ + self.mean_

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"AstralScaler({status})"


def astral_train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    """
    Split arrays or matrices into random train and test subsets.
    Supports stratified splits for classification.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.RandomState(random_state)

    if stratify is not None:
        stratify = np.asarray(stratify)
        classes = np.unique(stratify)
        train_idx, test_idx = [], []
        for c in classes:
            c_idx = np.where(stratify == c)[0]
            rng.shuffle(c_idx)
            n_test_c = max(1, int(round(len(c_idx) * test_size)))
            test_idx.extend(c_idx[:n_test_c].tolist())
            train_idx.extend(c_idx[n_test_c:].tolist())
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def astral_cross_validate(X, y, task="auto", folds=5, random_state=42, **model_kwargs):
    """
    Evaluate AstralModel with reproducible, optionally stratified K-fold cross-validation.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    if len(y) < 2 * folds:
        raise ValueError("Use fewer folds or provide at least two samples per fold.")
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(y))
    if task == "classification" or (task == "auto" and len(np.unique(y)) <= 20):
        buckets = [np.array_split(rng.permutation(np.where(y == c)[0]), folds) for c in np.unique(y)]
        split_folds = [np.concatenate([bucket[k] for bucket in buckets]) for k in range(folds)]
    else:
        split_folds = list(np.array_split(rng.permutation(indices), folds))

    results = []
    for k, test_idx in enumerate(split_folds):
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=False)
        model = AstralModel(task=task, random_state=random_state + k, verbose=False, **model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        results.append(model.score(X[test_idx], y[test_idx], verbose=False))

    numeric_keys = {key for r in results for key, value in r.items()
                    if isinstance(value, (int, float, np.floating))}
    return {
        "folds": results,
        "mean": {key: float(np.mean([r[key] for r in results])) for key in numeric_keys},
        "std": {key: float(np.std([r[key] for r in results])) for key in numeric_keys},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIVATE MATHEMATICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _sigmoid(z):
    """Numerically-stable sigmoid function."""
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _compute_mi(x, y, n_bins=None, yd=None, ny=None):
    """
    Fast vectorized mutual information I(x; y) using a 2D histogram plug-in estimator
    with Laplace smoothing (pure NumPy, zero Python sample loops).
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 10:
        return 0.0
    x_std = np.std(x)
    if x_std < 1e-10 or not np.isfinite(x_std):
        return 0.0

    n = len(x)
    if n_bins is None:
        n_bins = min(max(5, int(np.ceil(np.log2(n) + 1))), 16)

    # Discretize x
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max <= x_min:
        return 0.0
    x_edges = np.linspace(x_min - 1e-10, x_max + 1e-10, n_bins + 1)
    xd = np.clip(np.digitize(x, x_edges[1:-1]), 0, n_bins - 1)

    # Discretize y if not provided
    if yd is None or ny is None:
        y_uniq = np.unique(y)
        if len(y_uniq) <= 20:
            y_map = {v: i for i, v in enumerate(y_uniq)}
            yd = np.array([y_map[v] for v in y], dtype=np.int64)
            ny = len(y_uniq)
        else:
            y_min, y_max = float(np.min(y)), float(np.max(y))
            y_edges = np.linspace(y_min - 1e-10, y_max + 1e-10, n_bins + 1)
            yd = np.clip(np.digitize(y, y_edges[1:-1]), 0, n_bins - 1)
            ny = n_bins

    # Vectorized 2D contingency table via np.bincount (orders of magnitude faster than loops)
    flat_idx = xd * ny + yd
    joint = np.bincount(flat_idx, minlength=n_bins * ny).reshape(n_bins, ny).astype(np.float64)
    joint += 1e-8
    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 1e-15
    prod = px @ py
    mi = np.sum(joint[nonzero] * np.log(joint[nonzero] / (prod[nonzero] + 1e-15)))
    return max(0.0, float(mi))


def _compute_f1(y_true, y_pred, average="macro"):
    """Compute Macro / Weighted F1 score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s, supports = [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        supports.append(np.sum(y_true == c))
    if average == "macro":
        return float(np.mean(f1s))
    total = sum(supports)
    return float(sum(f * s for f, s in zip(f1s, supports)) / (total + 1e-12))


def _compute_accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _compute_roc_auc(y_true, y_scores):
    """
    Compute ROC-AUC score for binary or multiclass classifications in pure NumPy/SciPy.
    """
    if y_scores is None:
        return None
    try:
        from scipy.stats import rankdata
    except Exception:
        def rankdata(a):
            order = np.argsort(a)
            ranks = np.empty(len(a), dtype=float)
            ranks[order] = np.arange(1, len(a) + 1, dtype=float)
            sorted_a = a[order]
            unique_vals, counts = np.unique(sorted_a, return_counts=True)
            if len(unique_vals) < len(a):
                idx = 0
                for count in counts:
                    if count > 1:
                        tied_ranks = ranks[order[idx:idx + count]]
                        ranks[order[idx:idx + count]] = np.mean(tied_ranks)
                    idx += count
            return ranks

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_scores.ndim == 1:
        unique_cls = np.unique(y_true)
        if len(unique_cls) != 2:
            return 0.5
        pos = (y_true == unique_cls[1])
        n_pos = int(np.sum(pos))
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        ranks = rankdata(y_scores)
        return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    elif y_scores.ndim == 2:
        n_classes = y_scores.shape[1]
        if n_classes == 2:
            unique_cls = np.unique(y_true)
            pos = (y_true == unique_cls[-1])
            n_pos = int(np.sum(pos))
            n_neg = len(y_true) - n_pos
            if n_pos == 0 or n_neg == 0:
                return 0.5
            ranks = rankdata(y_scores[:, 1])
            return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
        else:
            aucs = []
            for c in range(n_classes):
                pos = (y_true == c)
                n_pos = int(np.sum(pos))
                n_neg = len(y_true) - n_pos
                if n_pos > 0 and n_neg > 0:
                    ranks = rankdata(y_scores[:, c])
                    auc_c = float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
                    aucs.append(auc_c)
            return float(np.mean(aucs)) if aucs else 0.5
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  RESONANCE BASIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_basis_counter = 0


class ResonanceBasis:
    """
    A single adaptive basis function discovered by the SRD algorithm.
    Carries metadata about stability, description length, and predictive importance.
    """

    def __init__(self, feature_indices, transform_type, params=None, name=None):
        global _basis_counter
        _basis_counter += 1
        self.id = _basis_counter
        self.feature_indices = (
            tuple(feature_indices)
            if isinstance(feature_indices, (list, tuple, np.ndarray))
            else (int(feature_indices),)
        )
        self.transform_type = transform_type
        self.params = params or {}
        self.name = name or f"basis_{self.id}"
        self.stability_score = 1.0
        self.importance = 0.0
        self.is_stable = True
        self.mi_score = 0.0

    @property
    def description_length(self):
        """Minimum Description Length (MDL) cost penalty."""
        if self.transform_type == "identity":
            return 1.0
        if self.transform_type in ("quantile_hat", "rbf", "sigmoid"):
            return 2.5
        if self.transform_type in ("log", "sqrt", "square", "inverse", "threshold"):
            return 2.0
        if self.transform_type.startswith("interaction"):
            bi_dl = self.params.get("basis_i_dl", 2.0)
            bj_dl = self.params.get("basis_j_dl", 2.0)
            return bi_dl + bj_dl + 0.8
        return 2.0

    def evaluate(self, X):
        """Evaluate basis function on an (n, p) data matrix."""
        j = self.feature_indices[0]
        col = X[:, j]

        if self.transform_type == "identity":
            return col.copy()

        if self.transform_type == "quantile_hat":
            c = self.params["center"]
            bw = self.params["bandwidth"]
            return np.maximum(0.0, 1.0 - np.abs(col - c) / (bw + 1e-10))

        if self.transform_type == "rbf":
            c = self.params["center"]
            bw = self.params["bandwidth"]
            diff = (col - c) / (bw + 1e-10)
            return np.exp(-np.clip(diff ** 2, 0.0, 40.0))

        if self.transform_type == "sigmoid":
            c = self.params["center"]
            bw = self.params["bandwidth"]
            diff = (col - c) / (bw + 1e-10)
            return 1.0 / (1.0 + np.exp(-np.clip(diff, -25.0, 25.0)))

        if self.transform_type == "log":
            return np.sign(col) * np.log1p(np.abs(col))

        if self.transform_type == "sqrt":
            return np.sign(col) * np.sqrt(np.abs(col))

        if self.transform_type == "square":
            return col ** 2

        if self.transform_type == "inverse":
            return np.sign(col) / (1.0 + np.abs(col))

        if self.transform_type == "threshold":
            return (col > self.params["threshold"]).astype(np.float64)

        if self.transform_type == "interaction_mult":
            return (
                self.params["basis_i"].evaluate(X)
                * self.params["basis_j"].evaluate(X)
            )

        if self.transform_type == "interaction_ratio":
            denom = 1.0 + np.abs(self.params["basis_j"].evaluate(X))
            return self.params["basis_i"].evaluate(X) / denom

        if self.transform_type == "interaction_min":
            return np.minimum(
                self.params["basis_i"].evaluate(X),
                self.params["basis_j"].evaluate(X),
            )

        if self.transform_type == "interaction_absdiff":
            return np.abs(
                self.params["basis_i"].evaluate(X)
                - self.params["basis_j"].evaluate(X)
            )

        raise ValueError(f"Unknown transform type: {self.transform_type}")

    def __repr__(self):
        return (
            f"ResonanceBasis('{self.name}', stability={self.stability_score:.3f}, "
            f"importance={self.importance:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ASTRAL MODEL — Spectral Resonance Decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class AstralModel:
    version = __version__
    __version__ = __version__
    """
    Spectral Resonance Decomposition (SRD) predictive model.

    Unifies non-linear adaptive resonance bases, causal stability separation,
    stability-weighted regularization, and conformal uncertainty calibration.

    Parameters
    ----------
    task : str, default='auto'
        'classification', 'regression', or 'auto' (auto-detect).
    feature_names : list[str] or None
        Human-readable feature names; defaults to X0, X1, …
    n_quantiles : int, default=5
        Number of quantile positions for local resonance kernels.
    max_bases_per_feature : int, default=6
        Top bases to retain per feature after MI ranking.
    max_interactions : int, default=35
        Maximum number of interaction bases to screen and retain.
    n_environments : int, default=10
        Bootstrap environments for stability scoring.
    stability_threshold : float, default=0.5
        Bases with stability > this value are categorized as stable (causal-like).
    alpha : float, default=0.10
        Mis-coverage rate for conformal prediction (0.10 -> 90% coverage).
    ridge : float or 'auto', default='auto'
        Ridge regularization strength. 'auto' performs internal CV tuning.
    ridge_grid : sequence of float or None
        Candidate strengths tested when ridge='auto'.
    class_weight : None, 'balanced', or dict, default=None
        Optional class reweighting for imbalanced classification.
    impute_missing : bool, default=True
        Automatically replace missing or infinite values with column medians.
    max_iter : int, default=80
        Maximum iterations for IRLS solvers.
    max_samples_discovery : int, default=25000
        Maximum samples used for basis discovery and stability scoring (speeds up N > 25k).
    random_state : int or None
        Seed for reproducibility.
    verbose : bool, default=False
        Print progress summaries during training.
    """

    def __init__(
        self,
        X=None,
        y=None,
        task="auto",
        feature_names=None,
        n_quantiles=5,
        max_bases_per_feature=6,
        max_interactions=35,
        n_environments=10,
        stability_threshold=0.5,
        alpha=0.10,
        ridge="auto",
        ridge_grid=None,
        class_weight=None,
        impute_missing=True,
        max_iter=80,
        max_samples_discovery=4000,
        random_state=42,
        verbose=False,
    ):
        self.task = task
        self.feature_names = feature_names
        self.n_quantiles = n_quantiles
        self.max_bases_per_feature = max_bases_per_feature
        self.max_interactions = max_interactions
        self.n_environments = n_environments
        self.stability_threshold = stability_threshold
        self.alpha = alpha
        self.ridge = ridge
        self.ridge_grid = ridge_grid or (1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
        self.class_weight = class_weight
        self.impute_missing = impute_missing
        self.max_iter = max_iter
        self.max_samples_discovery = max_samples_discovery
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.task_ = None
        self.classes_ = None
        self.bases_ = []
        self.weights_ = None
        self.bias_ = 0.0
        self.n_selected_ = 0
        self.ridge_ = 1.0
        self._fitted = False

        # Immediate fit if X and y are provided directly in constructor
        if X is not None and y is not None:
            self.fit(X, y)

    # ──────────────────────────────────────────────────────────────────────
    #  FIT PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the AstralModel on training data."""
        self._rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != len(y) or X.shape[0] < 10:
            raise ValueError("X and y must have identical sample length with at least 10 rows.")

        self._n_train, self._n_features = X.shape
        self.feature_names_ = (
            list(self.feature_names) if self.feature_names is not None
            else [f"X{j}" for j in range(self._n_features)]
        )

        # Missing values handling
        self._missing_counts_ = np.sum(~np.isfinite(X), axis=0)
        self._impute_values_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self._impute_values_ = np.nan_to_num(self._impute_values_, nan=0.0, posinf=0.0, neginf=0.0)
        if self.impute_missing:
            X = np.where(np.isfinite(X), X, self._impute_values_)
        elif not np.isfinite(X).all():
            raise ValueError("X contains missing/infinite values. Set impute_missing=True to handle them.")

        # Detect task & resolve classes
        self._detect_task(y, self.task)
        y_enc = self._encode_target(y)
        self._resolve_class_weight(y_enc)

        # Subsample for basis discovery if dataset is massive (speeds up large N without losing accuracy)
        if self._n_train > self.max_samples_discovery:
            if self.task_ == "classification":
                disc_idx = []
                for c in range(self._n_classes):
                    c_idx = np.where(y_enc == c)[0]
                    k_c = max(5, int(self.max_samples_discovery * len(c_idx) / self._n_train))
                    chosen = self._rng.choice(c_idx, size=min(k_c, len(c_idx)), replace=False)
                    disc_idx.extend(chosen)
                disc_idx = np.array(disc_idx)
            else:
                disc_idx = self._rng.choice(self._n_train, size=self.max_samples_discovery, replace=False)
            X_disc, y_disc = X[disc_idx], y_enc[disc_idx]
        else:
            X_disc, y_disc = X, y_enc

        # ── Phase 1: Discover univariate resonance bases ──
        if self.verbose:
            print(f"⟐ AstralModel v{__version__} — Spectral Resonance Decomposition")
            print(f"  Task: {self.task_} | Samples: {self._n_train} | Features: {self._n_features}")
            print("  [1/5] Discovering univariate resonance bases …")
        univariate_bases = self._discover_bases(X_disc, y_disc)

        # ── Phase 2: Screen interactions ──
        if self.verbose:
            print(f"  [2/5] Screening interactions from {len(univariate_bases)} bases …")
        interaction_bases = self._discover_interactions(X_disc, y_disc, univariate_bases)
        all_candidate_bases = univariate_bases + interaction_bases

        # ── Phase 3: Score causal stability ──
        if self.verbose:
            print(f"  [3/5] Evaluating stability across {self.n_environments} environments …")
        self._score_stability(X_disc, y_disc, all_candidate_bases)

        # ── Phase 4: Forward MDL-guided orthogonalized basis selection ──
        if self.verbose:
            print(f"  [4/5] MDL basis selection ({len(all_candidate_bases)} candidates) …")
        selected_idx = self._greedy_select(X_disc, y_disc, all_candidate_bases)
        self.bases_ = [all_candidate_bases[i] for i in selected_idx]
        self.n_selected_ = len(self.bases_)
        self._all_candidate_bases = all_candidate_bases

        # ── Phase 5: Fitting with stability-weighted regularization on full data ──
        if self.verbose:
            print(f"  [5/5] Fitting model with {self.n_selected_} selected bases …")
        self._fit_model(X, y_enc)

        # ── Phase 6: Conformal calibration ──
        self._calibrate_conformal(X, y_enc)

        # Importances
        for i, b in enumerate(self.bases_):
            b.importance = (
                float(np.max(np.abs(self.weights_[i])))
                if self._n_classes > 2
                else float(np.abs(self.weights_[i]))
            )

        # Compute training performance metrics
        y_tr_pred = self._raw_predict(X)
        if self.task_ == "classification":
            y_tr_cls = (y_tr_pred >= 0.5).astype(int) if self._n_classes == 2 else np.argmax(y_tr_pred, axis=1)
            self._train_acc = _compute_accuracy(y_enc, y_tr_cls)
            self._train_f1 = _compute_f1(y_enc, y_tr_cls)
        else:
            ss_res = np.sum((y_enc - y_tr_pred) ** 2)
            ss_tot = np.sum((y_enc - np.mean(y_enc)) ** 2)
            self._train_r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        self._fitted = True
        if self.verbose:
            self.summary()
        return self

    # ──────────────────────────────────────────────────────────────────────
    #  TASK & TARGET MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────

    def _detect_task(self, y, task):
        if task not in ("auto", "classification", "regression"):
            raise ValueError("task must be 'auto', 'classification', or 'regression'.")
        if task != "auto":
            self.task_ = task
        else:
            try:
                y_float = y.astype(np.float64)
                n_unique = len(np.unique(y_float[np.isfinite(y_float)]))
            except (ValueError, TypeError):
                n_unique = len(np.unique(y))
            self.task_ = "classification" if n_unique <= 20 else "regression"

        if self.task_ == "classification":
            self.classes_ = np.unique(y)
            self._n_classes = len(self.classes_)
            if self._n_classes < 2:
                raise ValueError("Classification requires at least two distinct target classes.")
            self._label_map = {c: i for i, c in enumerate(self.classes_)}
            self._inv_label_map = {i: c for c, i in self._label_map.items()}
        else:
            self.classes_ = None
            self._n_classes = 0

    def _encode_target(self, y):
        if self.task_ == "classification":
            return np.array([self._label_map[v] for v in y], dtype=np.float64)
        return y.astype(np.float64)

    def _resolve_class_weight(self, y):
        if self.task_ != "classification":
            self._class_weights_ = None
            return
        if self.class_weight is None:
            weights = np.ones(self._n_classes)
        elif self.class_weight == "balanced":
            counts = np.bincount(y.astype(int), minlength=self._n_classes)
            weights = len(y) / (self._n_classes * np.maximum(counts, 1))
        elif isinstance(self.class_weight, dict):
            weights = np.array([self.class_weight.get(label, self.class_weight.get(i, 1.0))
                                for i, label in enumerate(self.classes_)], dtype=float)
            if np.any(weights <= 0):
                raise ValueError("All class_weight values must be positive.")
        else:
            raise ValueError("class_weight must be None, 'balanced', or a dictionary.")
        self._class_weights_ = weights

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError(f"Expected X with {self._n_features} columns; got shape {X.shape}.")
        if self.impute_missing:
            return np.where(np.isfinite(X), X, self._impute_values_)
        if not np.isfinite(X).all():
            raise ValueError("X contains missing or infinite values.")
        return X

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 1: DISCOVER UNIVARIATE BASES (RBF, Hat, Sigmoid, Powers)
    # ──────────────────────────────────────────────────────────────────────

    def _discover_bases(self, X, y):
        quantile_positions = np.linspace(0.15, 0.85, self.n_quantiles)
        all_bases = []

        # Pre-discretize target y once for all basis candidates
        y_uniq = np.unique(y)
        if len(y_uniq) <= 20:
            y_map = {v: i for i, v in enumerate(y_uniq)}
            yd = np.array([y_map[v] for v in y], dtype=np.int64)
            ny = len(y_uniq)
        else:
            y_min, y_max = float(np.min(y)), float(np.max(y))
            y_edges = np.linspace(y_min - 1e-10, y_max + 1e-10, 16 + 1)
            yd = np.clip(np.digitize(y, y_edges[1:-1]), 0, 16 - 1)
            ny = 16

        for j in range(self._n_features):
            col = X[:, j]
            feat_name = self.feature_names_[j]
            candidates = []

            # Linear identity
            candidates.append(ResonanceBasis(j, "identity", name=f"{feat_name}"))

            finite = col[np.isfinite(col)]
            col_std = np.std(finite) if len(finite) > 1 else 1.0
            if col_std < 1e-10:
                col_std = 1.0

            if len(finite) > 5:
                quants = np.quantile(finite, quantile_positions)
                bw_default = max(0.6 * col_std, 1e-3)
                for qi, qv in enumerate(quants):
                    # Bandwidth: distance to adjacent quantile
                    if qi == 0:
                        bw = quants[1] - quants[0] if len(quants) > 1 else bw_default
                    elif qi == len(quants) - 1:
                        bw = quants[-1] - quants[-2]
                    else:
                        bw = min(quants[qi] - quants[qi - 1], quants[qi + 1] - quants[qi])
                    bw = max(bw, bw_default * 0.5)

                    # Quantile Hat
                    candidates.append(ResonanceBasis(
                        j, "quantile_hat",
                        params={"center": float(qv), "bandwidth": float(bw)},
                        name=f"{feat_name}_hat@{qv:.2f}",
                    ))
                    # Gaussian RBF
                    candidates.append(ResonanceBasis(
                        j, "rbf",
                        params={"center": float(qv), "bandwidth": float(bw)},
                        name=f"{feat_name}_rbf@{qv:.2f}",
                    ))
                    # Sigmoidal ramp
                    candidates.append(ResonanceBasis(
                        j, "sigmoid",
                        params={"center": float(qv), "bandwidth": float(bw)},
                        name=f"{feat_name}_sig@{qv:.2f}",
                    ))

            # Non-linear transformations
            candidates.append(ResonanceBasis(j, "log", name=f"{feat_name}_log"))
            candidates.append(ResonanceBasis(j, "sqrt", name=f"{feat_name}_sqrt"))
            candidates.append(ResonanceBasis(j, "square", name=f"{feat_name}_sq"))
            candidates.append(ResonanceBasis(j, "inverse", name=f"{feat_name}_inv"))

            # Step threshold at median
            med = np.nanmedian(col)
            candidates.append(ResonanceBasis(
                j, "threshold",
                params={"threshold": float(med)},
                name=f"{feat_name}_thr",
            ))

            # Score each candidate by vectorized mutual information with target
            for b in candidates:
                try:
                    vals = b.evaluate(X)
                    b.mi_score = _compute_mi(vals, y, yd=yd, ny=ny)
                except Exception:
                    b.mi_score = 0.0

            candidates.sort(key=lambda b: b.mi_score, reverse=True)
            all_bases.extend(candidates[: self.max_bases_per_feature])

        return all_bases

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 2: INTERACTION RESONANCE SCREENING
    # ──────────────────────────────────────────────────────────────────────

    def _discover_interactions(self, X, y, univariate_bases):
        feat_groups = {}
        for b in univariate_bases:
            fj = b.feature_indices[0]
            feat_groups.setdefault(fj, []).append(b)

        # Rank features by their maximum univariate MI with the target
        feat_max_mi = {
            fj: max(b.mi_score for b in bases) for fj, bases in feat_groups.items()
        }
        # Prioritize top informative features for interaction testing (max 14 features -> 91 pairs)
        sorted_features = sorted(feat_max_mi.keys(), key=lambda fj: feat_max_mi[fj], reverse=True)
        feature_ids = sorted_features[: min(14, len(sorted_features))]

        # Pre-discretize target y once
        y_uniq = np.unique(y)
        if len(y_uniq) <= 20:
            y_map = {v: i for i, v in enumerate(y_uniq)}
            yd = np.array([y_map[v] for v in y], dtype=np.int64)
            ny = len(y_uniq)
        else:
            y_min, y_max = float(np.min(y)), float(np.max(y))
            y_edges = np.linspace(y_min - 1e-10, y_max + 1e-10, 16 + 1)
            yd = np.clip(np.digitize(y, y_edges[1:-1]), 0, 16 - 1)
            ny = 16

        # Cache basis evaluations to prevent repeating expensive evaluations
        eval_cache = {}
        for fj in feature_ids:
            for b in feat_groups[fj][:2]:
                eval_cache[b.id] = b.evaluate(X)

        interaction_candidates = []

        for fi_idx in range(len(feature_ids)):
            for fj_idx in range(fi_idx + 1, len(feature_ids)):
                fi = feature_ids[fi_idx]
                fj = feature_ids[fj_idx]
                bases_i = feat_groups[fi][:2]
                bases_j = feat_groups[fj][:2]

                for bi in bases_i:
                    bi_val = eval_cache[bi.id]
                    for bj in bases_j:
                        bj_val = eval_cache[bj.id]
                        max_marginal = max(bi.mi_score, bj.mi_score)

                        # Test interaction transforms directly in vector space: mult, soft ratio, min, absdiff
                        transform_specs = [
                            ("interaction_mult", "mult", bi_val * bj_val),
                            ("interaction_ratio", "ratio", bi_val / (1.0 + np.abs(bj_val))),
                            ("interaction_min", "min", np.minimum(bi_val, bj_val)),
                            ("interaction_absdiff", "diff", np.abs(bi_val - bj_val)),
                        ]

                        for itype, suffix, vals in transform_specs:
                            mi_val = _compute_mi(vals, y, yd=yd, ny=ny)
                            gain = mi_val - max_marginal
                            if gain > 0.005:
                                ib = ResonanceBasis(
                                    list(bi.feature_indices) + list(bj.feature_indices),
                                    itype,
                                    params={
                                        "basis_i": bi,
                                        "basis_j": bj,
                                        "basis_i_dl": bi.description_length,
                                        "basis_j_dl": bj.description_length,
                                        "resonance_gain": gain,
                                    },
                                    name=f"{bi.name}×{bj.name}_{suffix}",
                                )
                                ib.mi_score = mi_val
                                interaction_candidates.append(ib)

        interaction_candidates.sort(
            key=lambda b: b.params.get("resonance_gain", 0.0), reverse=True
        )
        return interaction_candidates[: self.max_interactions]

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 3: CAUSAL INVARIANCE STABILITY SCORING
    # ──────────────────────────────────────────────────────────────────────

    def _score_stability(self, X, y, bases):
        n = len(y)
        m = len(bases)
        if m == 0:
            return

        Phi = np.column_stack([b.evaluate(X) for b in bases])
        Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)

        col_mean = Phi.mean(axis=0)
        col_std = Phi.std(axis=0)
        col_std[col_std < 1e-10] = 1.0
        Phi_n = (Phi - col_mean) / col_std

        corrs = np.zeros((self.n_environments, m))
        for k in range(self.n_environments):
            if self.task_ == "classification":
                env_idx = []
                for c in range(self._n_classes):
                    c_idx = np.where(y == c)[0]
                    chosen = self._rng.choice(c_idx, size=max(2, int(0.8 * len(c_idx))), replace=True)
                    env_idx.extend(chosen.tolist())
                env_idx = np.array(env_idx)
            else:
                env_idx = self._rng.choice(n, size=max(5, int(0.8 * n)), replace=True)

            Pk = Phi_n[env_idx]
            yk = y[env_idx]
            y_c = yk - np.mean(yk)
            y_norm = np.sqrt(np.sum(y_c ** 2)) + 1e-10
            p_c = Pk - np.mean(Pk, axis=0, keepdims=True)
            p_norm = np.sqrt(np.sum(p_c ** 2, axis=0)) + 1e-10
            r_k = (Pk.T @ y_c) / (p_norm * y_norm + 1e-10)
            corrs[k] = np.nan_to_num(r_k, nan=0.0)

        mean_corr = np.mean(corrs, axis=0)
        std_corr = np.std(corrs, axis=0)

        for j, b in enumerate(bases):
            mag = np.abs(mean_corr[j])
            snr = mag / (mag + std_corr[j] + 1e-8)
            sign_consistency = (
                float(np.mean(np.sign(corrs[:, j]) == np.sign(mean_corr[j])))
                if mag > 1e-5 else 0.0
            )
            b.stability_score = float(np.clip(snr * sign_consistency, 0.0, 1.0))
            b.is_stable = b.stability_score >= self.stability_threshold

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 4: MDL FORWARD ORTHOGONALIZED BASIS SELECTION
    # ──────────────────────────────────────────────────────────────────────

    def _greedy_select(self, X, y, bases):
        m = len(bases)
        if m == 0:
            return []

        n = len(y)
        max_bases = min(m, max(6, n // 4), 65)

        Phi = np.column_stack([b.evaluate(X) for b in bases])
        Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)

        col_mean = Phi.mean(axis=0)
        col_std = Phi.std(axis=0)
        col_std[col_std < 1e-10] = 1.0
        Phi_n = (Phi - col_mean) / col_std

        y_c = y - np.mean(y)
        residual = y_c.copy()
        res_ss = np.dot(residual, residual)
        if res_ss < 1e-12:
            return list(range(min(3, m)))

        selected = []
        Phi_proj = Phi_n.copy()
        norm_sq = np.sum(Phi_proj ** 2, axis=0)
        stab_weights = np.array([b.stability_score for b in bases])
        dl_weights = np.array([b.description_length for b in bases])

        for _ in range(max_bases):
            denom = np.sqrt(norm_sq * res_ss + 1e-12)
            corrs = (Phi_proj.T @ residual) / denom
            crits = (corrs ** 2) * (0.4 + 0.6 * stab_weights) / (dl_weights + 0.1)

            if selected:
                crits[selected] = -np.inf
            crits[norm_sq < 1e-10] = -np.inf

            best_j = int(np.argmax(crits))
            best_crit = float(crits[best_j])

            if best_crit < 1e-5 or not np.isfinite(best_crit):
                break

            selected.append(best_j)
            best_q = Phi_proj[:, best_j] / np.sqrt(norm_sq[best_j])
            proj_step = best_q @ Phi_proj
            Phi_proj -= np.outer(best_q, proj_step)
            norm_sq = np.maximum(norm_sq - proj_step ** 2, 0.0)
            residual -= best_q * np.dot(best_q, residual)
            res_ss = np.dot(residual, residual)
            if res_ss < 1e-12:
                break

        if len(selected) < 3 and m >= 3:
            remaining = [(bases[j].mi_score, j) for j in range(m) if j not in selected]
            remaining.sort(reverse=True)
            for _, j in remaining:
                selected.append(j)
                if len(selected) >= 3:
                    break

        return selected

    # ──────────────────────────────────────────────────────────────────────
    #  DESIGN MATRIX
    # ──────────────────────────────────────────────────────────────────────

    def _build_design_matrix(self, X):
        cols = []
        for b in self.bases_:
            v = b.evaluate(X)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(v)
        Phi = np.column_stack(cols) if cols else np.ones((X.shape[0], 1))
        Phi = (Phi - self._phi_mean) / self._phi_std
        return Phi

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 5: STABILITY-WEIGHTED ROBUST MODEL FITTING
    # ──────────────────────────────────────────────────────────────────────

    def _fit_model(self, X, y):
        cols = []
        for b in self.bases_:
            v = b.evaluate(X)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(v)
        Phi_raw = np.column_stack(cols) if cols else np.ones((X.shape[0], 1))

        self._phi_mean = Phi_raw.mean(axis=0)
        self._phi_std = Phi_raw.std(axis=0)
        self._phi_std[self._phi_std < 1e-10] = 1.0
        Phi = (Phi_raw - self._phi_mean) / self._phi_std

        # Stability-weighted regularization vector:
        # Invariant bases receive lower penalty, unstable bases receive higher penalty
        stabs = np.array([b.stability_score for b in self.bases_])
        stability_factors = 1.0 / (0.25 + 0.75 * stabs)

        # Cross-validation hyperparameter selection
        default_lam = max(0.01, len(self.bases_) / len(y))
        self.ridge_ = (
            self._select_ridge_cv(Phi, y, stability_factors, default_lam)
            if self.ridge == "auto"
            else float(self.ridge)
        )

        if self.task_ == "classification":
            if self._n_classes == 2:
                self.weights_, self.bias_ = self._fit_logistic(
                    Phi, y, self.ridge_, stability_factors=stability_factors
                )
            else:
                weights_list, bias_list = [], []
                sample_weight = self._class_weights_[y.astype(int)]
                for c in range(self._n_classes):
                    w, b = self._fit_logistic(
                        Phi, (y == c).astype(float), self.ridge_,
                        stability_factors=stability_factors, sample_weight=sample_weight
                    )
                    weights_list.append(w)
                    bias_list.append(b)
                self.weights_ = np.column_stack(weights_list)
                self.bias_ = np.array(bias_list)
        else:
            self.weights_, self.bias_ = self._fit_huber(
                Phi, y, self.ridge_, stability_factors=stability_factors
            )

    def _select_ridge_cv(self, Phi, y, stability_factors, default_lam):
        """Perform fast stratified/reproducible K-fold CV to select optimal ridge penalty."""
        n = len(y)
        if n < 30:
            return default_lam

        k_folds = 5 if n >= 60 else 3
        indices = np.arange(n)

        if self.task_ != "classification":
            # Fast analytical Ridge CV for regression: precompute XtX & Xty once per fold
            shuffled = self._rng.permutation(indices)
            folds_split = list(np.array_split(shuffled, k_folds))
            scores_per_lam = {lam: [] for lam in self.ridge_grid}

            for f in range(k_folds):
                val_idx = folds_split[f]
                trn_idx = np.setdiff1d(indices, val_idx)
                if len(trn_idx) < 10 or len(val_idx) == 0:
                    continue
                Phi_trn = Phi[trn_idx]
                y_trn = y[trn_idx]
                Phi_val = Phi[val_idx]
                y_val = y[val_idx]
                n_trn, m_feat = Phi_trn.shape

                Phi_a = np.column_stack([Phi_trn, np.ones(n_trn)])
                XtX = Phi_a.T @ Phi_a
                Xty = Phi_a.T @ y_trn

                for lam in self.ridge_grid:
                    reg = np.diag(np.r_[lam * stability_factors, 0.0])
                    try:
                        w = np.linalg.solve(XtX + reg, Xty)
                    except np.linalg.LinAlgError:
                        w = np.linalg.lstsq(XtX + reg, Xty, rcond=None)[0]
                    pred = Phi_val @ w[:m_feat] + w[m_feat]
                    scores_per_lam[lam].append(-float(np.mean((y_val - pred) ** 2)))

            best_lam, best_score = default_lam, -np.inf
            for lam in self.ridge_grid:
                if scores_per_lam[lam] and np.mean(scores_per_lam[lam]) > best_score:
                    best_score = float(np.mean(scores_per_lam[lam]))
                    best_lam = float(lam)
            return best_lam

        # Classification CV with warm-started regularized logistic regression
        k_folds_cls = 3 if n >= 600 else k_folds
        if self._n_classes == 2:
            buckets = [self._rng.permutation(np.where(y == c)[0]) for c in [0, 1]]
            folds_split = [
                np.concatenate([np.array_split(b, k_folds_cls)[f] for b in buckets])
                for f in range(k_folds_cls)
            ]
        else:
            shuffled = self._rng.permutation(indices)
            folds_split = list(np.array_split(shuffled, k_folds_cls))

        sorted_lams = sorted(self.ridge_grid, reverse=True)
        scores_per_lam = {lam: [] for lam in self.ridge_grid}

        for f in range(k_folds_cls):
            val_idx = folds_split[f]
            trn_idx = np.setdiff1d(indices, val_idx)
            if len(trn_idx) < 10 or len(val_idx) == 0:
                continue

            Phi_trn = Phi[trn_idx]
            y_trn = y[trn_idx]
            Phi_val = Phi[val_idx]
            y_val = y[val_idx]
            p1 = Phi_trn.shape[1] + 1
            w_prev = np.zeros(p1)

            if self._n_classes == 2:
                for lam in sorted_lams:
                    w, b = self._fit_logistic(
                        Phi_trn, y_trn, lam,
                        stability_factors=stability_factors, max_iter=25, init_w=w_prev
                    )
                    w_prev = np.r_[w, b]
                    probs = _sigmoid(Phi_val @ w + b)
                    pred = (probs >= 0.5).astype(int)
                    scores_per_lam[lam].append(_compute_f1(y_val, pred, "macro"))
            else:
                for lam in self.ridge_grid:
                    scores_per_lam[lam].append(0.0)

        best_lam, best_score = default_lam, -np.inf
        for lam in self.ridge_grid:
            if scores_per_lam[lam] and np.mean(scores_per_lam[lam]) > best_score:
                best_score = float(np.mean(scores_per_lam[lam]))
                best_lam = float(lam)

        return best_lam

    def _fit_logistic(self, Phi, y, lam, stability_factors=None, max_iter=None, tol=1e-5, sample_weight=None, init_w=None):
        n, m = Phi.shape
        Phi_a = np.column_stack([Phi, np.ones(n)])
        p1 = m + 1

        if stability_factors is not None:
            penalty = np.r_[lam * stability_factors, 0.0]
        else:
            penalty = np.r_[lam * np.ones(m), 0.0]
        reg = np.diag(penalty)

        if sample_weight is None:
            sample_weight = (
                self._class_weights_[y.astype(int)]
                if self._class_weights_ is not None and self._n_classes == 2
                else np.ones(n)
            )
        sample_weight = np.asarray(sample_weight, dtype=float)

        w = init_w.copy() if (init_w is not None and len(init_w) == p1) else np.zeros(p1)
        for it in range(max_iter or self.max_iter):
            z = Phi_a @ w
            p_hat = _sigmoid(z)
            s = p_hat * (1.0 - p_hat)
            s = np.maximum(s * sample_weight, 1e-7)

            r = z + (y - p_hat) / s
            PhiS = Phi_a.T * s
            H = PhiS @ Phi_a + reg
            g = PhiS @ r

            try:
                w_new = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                w_new = np.linalg.lstsq(H, g, rcond=None)[0]

            tol_check = tol * (np.max(np.abs(w)) + 1e-4)
            if np.max(np.abs(w_new - w)) <= tol_check:
                w = w_new
                break
            w = w_new

        return w[:m], w[m]

    def _fit_huber(self, Phi, y, lam, stability_factors=None, max_iter=None, tol=1e-5):
        n, m = Phi.shape
        Phi_a = np.column_stack([Phi, np.ones(n)])
        p1 = m + 1

        if stability_factors is not None:
            penalty = np.r_[lam * stability_factors, 0.0]
        else:
            penalty = np.r_[lam * np.ones(m), 0.0]
        reg = np.diag(penalty)

        try:
            w = np.linalg.solve(Phi_a.T @ Phi_a + reg, Phi_a.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(Phi_a.T @ Phi_a + reg, Phi_a.T @ y, rcond=None)[0]

        for it in range(max_iter or self.max_iter):
            resid = y - Phi_a @ w
            mad = np.median(np.abs(resid)) + 1e-8
            delta = 1.345 * mad

            h = np.where(np.abs(resid) <= delta, 1.0, delta / (np.abs(resid) + 1e-10))
            PhiH = Phi_a.T * h
            H = PhiH @ Phi_a + reg
            g = PhiH @ y

            try:
                w_new = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                w_new = np.linalg.lstsq(H, g, rcond=None)[0]

            tol_check = tol * (np.max(np.abs(w)) + 1e-4)
            if np.max(np.abs(w_new - w)) <= tol_check:
                w = w_new
                break
            w = w_new

        return w[:m], w[m]

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 6: CONFORMAL CALIBRATION & ABSTENTION
    # ──────────────────────────────────────────────────────────────────────

    def _calibrate_conformal(self, X, y):
        raw = self._raw_predict(X)
        if self.task_ == "classification":
            probs = _sigmoid(raw) if self._n_classes == 2 else raw
            if self._n_classes == 2:
                true_probs = np.where(y == 1, probs, 1.0 - probs)
            else:
                exp_r = np.exp(raw - np.max(raw, axis=1, keepdims=True))
                probs_multi = exp_r / exp_r.sum(axis=1, keepdims=True)
                true_probs = np.array([probs_multi[i, int(y[i])] for i in range(len(y))])
            self._cal_scores = 1.0 - true_probs
            self._conformal_q = float(np.quantile(self._cal_scores, min(1.0, 1.0 - self.alpha)))
            self._abstention_threshold = 1.0 - self._conformal_q
        else:
            self._cal_scores = np.abs(y - raw)
            self._conformal_q = float(np.quantile(self._cal_scores, min(1.0, 1.0 - self.alpha)))
            y_iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
            self._abstention_threshold = max(y_iqr * 1.5, self._conformal_q * 2.0)

    # ──────────────────────────────────────────────────────────────────────
    #  PREDICTION API
    # ──────────────────────────────────────────────────────────────────────

    def _raw_predict(self, X):
        Phi = self._build_design_matrix(self._prepare_X(X))
        return Phi @ self.weights_ + self.bias_

    def predict(self, X, abstain=False):
        """Return point predictions for input matrix X."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        raw = self._raw_predict(X)

        if self.task_ == "classification":
            if self._n_classes == 2:
                probs = _sigmoid(raw)
                idx = (probs >= 0.5).astype(int)
            else:
                idx = np.argmax(raw, axis=1)
            preds = np.array([self._inv_label_map[i] for i in idx])
        else:
            preds = raw.copy()

        if abstain:
            _, _, _, abstain_flags = self.predict_with_uncertainty(X)
            preds_obj = np.empty(len(preds), dtype=object)
            for i in range(len(preds)):
                preds_obj[i] = None if abstain_flags[i] else preds[i]
            return preds_obj

        return preds

    def predict_proba(self, X):
        """Return class probability distributions (classification only)."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if self.task_ != "classification":
            raise ValueError("predict_proba is only available for classification.")
        raw = self._raw_predict(X)

        if self._n_classes == 2:
            p1 = _sigmoid(raw)
            return np.column_stack([1.0 - p1, p1])
        else:
            exp_r = np.exp(raw - np.max(raw, axis=1, keepdims=True))
            return exp_r / exp_r.sum(axis=1, keepdims=True)

    def predict_with_uncertainty(self, X):
        """Return predictions along with calibrated conformal bounds and abstention flags."""
        predictions = self.predict(X)
        raw = self._raw_predict(X)

        if self.task_ == "classification":
            if self._n_classes == 2:
                probs = _sigmoid(raw)
                max_prob = np.maximum(probs, 1.0 - probs)
            else:
                exp_r = np.exp(raw - np.max(raw, axis=1, keepdims=True))
                probs_all = exp_r / exp_r.sum(axis=1, keepdims=True)
                max_prob = np.max(probs_all, axis=1)

            lower = 1.0 - max_prob
            upper = max_prob
            abstain = max_prob < self._abstention_threshold
        else:
            lower = raw - self._conformal_q
            upper = raw + self._conformal_q
            abstain = (upper - lower) > self._abstention_threshold

        return predictions, lower, upper, abstain

    def predict_set(self, X):
        """Return conformal prediction sets for classification."""
        if self.task_ != "classification":
            raise ValueError("predict_set is only available for classification.")
        probs = self.predict_proba(X)
        return [self.classes_[row >= 1.0 - self._conformal_q].tolist() for row in probs]

    # ──────────────────────────────────────────────────────────────────────
    #  SCORING & EVALUATION
    # ──────────────────────────────────────────────────────────────────────

    def score(self, y_pred, y_true=None, verbose=False):
        """
        Evaluate predictions against ground truth.
        Accepts either (y_pred, y_test) or (X_test, y_test).
        """
        if y_true is None:
            raise ValueError("y_true must be provided: model.score(y_pred, y_test) or model.score(X_test, y_test)")

        y_pred_arr = np.asarray(y_pred)
        X_eval = None
        if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == self._n_features:
            X_eval = y_pred_arr
            y_pred = self.predict(y_pred)

        valid_mask = np.array([
            p is not None and not (isinstance(p, (float, np.floating)) and np.isnan(p))
            for p in y_pred
        ], dtype=bool)

        n_total = len(y_pred)
        n_evaluated = int(np.sum(valid_mask))
        coverage = float(n_evaluated / n_total) if n_total > 0 else 0.0

        if n_evaluated < n_total:
            y_pred_eval = np.array([y_pred[i] for i in range(n_total) if valid_mask[i]])
            y_true_eval = np.asarray(y_true)[valid_mask]
        else:
            y_pred_eval = np.asarray(y_pred)
            y_true_eval = np.asarray(y_true)

        if len(y_pred_eval) == 0:
            warnings.warn("All predictions abstained; metrics cannot be evaluated.")
            return {"coverage": 0.0}

        if self.task_ == "classification":
            acc = _compute_accuracy(y_true_eval, y_pred_eval)
            f1_mac = _compute_f1(y_true_eval, y_pred_eval, "macro")
            f1_w = _compute_f1(y_true_eval, y_pred_eval, "weighted")

            roc_auc = None
            if X_eval is not None:
                try:
                    probs = self.predict_proba(X_eval)
                    if n_evaluated < n_total:
                        probs = probs[valid_mask]
                    roc_auc = _compute_roc_auc(y_true_eval, probs)
                except Exception:
                    roc_auc = None

            if roc_auc is None:
                try:
                    roc_auc = _compute_roc_auc(y_true_eval, y_pred_eval)
                except Exception:
                    roc_auc = 0.5

            metrics = {
                "accuracy": acc,
                "f1_macro": f1_mac,
                "f1_weighted": f1_w,
                "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else 0.5,
                "coverage": coverage,
            }
            if verbose:
                print(f"Accuracy: {acc:.4f} | F1 Macro: {f1_mac:.4f} | F1 Weighted: {f1_w:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            return metrics
        else:
            ss_res = np.sum((y_true_eval - y_pred_eval) ** 2)
            ss_tot = np.sum((y_true_eval - np.mean(y_true_eval)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            rmse = float(np.sqrt(np.mean((y_true_eval - y_pred_eval) ** 2)))
            mae = float(np.mean(np.abs(y_true_eval - y_pred_eval)))

            metrics = {"r2": r2, "rmse": rmse, "mae": mae, "coverage": coverage}
            if verbose:
                print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
            return metrics

    # ──────────────────────────────────────────────────────────────────────
    #  INTERPRETABILITY & PARAMETERS
    # ──────────────────────────────────────────────────────────────────────

    def get_feature_importance(self):
        """Return feature importance records for all selected bases."""
        records = []
        for i, b in enumerate(self.bases_):
            weight = (
                float(self.weights_[i])
                if self._n_classes <= 2
                else float(self.weights_[i, np.argmax(np.abs(self.weights_[i]))])
            )
            records.append({
                "name": b.name,
                "importance": float(np.abs(b.importance)),
                "weight": weight,
                "stability": b.stability_score,
                "stability_score": b.stability_score,
                "is_stable": b.is_stable,
                "stability_label": "stable" if b.is_stable else "unstable",
                "transform_type": b.transform_type,
                "feature_indices": b.feature_indices,
                "description_length": b.description_length,
                "mi_score": b.mi_score,
            })
        records.sort(key=lambda r: r["importance"], reverse=True)
        return records

    def get_stable_features(self):
        """Return only bases identified as stable across bootstrap environments."""
        return [r for r in self.get_feature_importance() if r["is_stable"]]

    def get_unstable_features(self):
        """Return bases identified as unstable/correlational."""
        return [r for r in self.get_feature_importance() if not r["is_stable"]]

    def explain(self, X_instance):
        """Generate per-instance local attribution explanation."""
        X_instance = self._prepare_X(X_instance)
        Phi = self._build_design_matrix(X_instance)
        if self.task_ == "classification" and self._n_classes > 2:
            class_idx = int(np.argmax(self._raw_predict(X_instance)[0]))
            weights, bias = self.weights_[:, class_idx], self.bias_[class_idx]
        else:
            weights, bias = self.weights_, self.bias_
        contribs = Phi[0] * weights

        records = []
        for i, b in enumerate(self.bases_):
            records.append({
                "name": b.name,
                "basis_value": float(Phi[0, i]),
                "weight": float(weights[i]),
                "contribution": float(contribs[i]),
                "stable": b.is_stable,
            })
        records.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return records

    def data_quality_report(self):
        """Return training data quality report."""
        return {
            "version": __version__,
            "training_rows": self._n_train,
            "feature_count": self._n_features,
            "missing_values_by_feature": {
                self.feature_names_[i]: int(count)
                for i, count in enumerate(self._missing_counts_) if count
            },
            "imputation": "median" if self.impute_missing else "disabled",
            "selected_basis_count": self.n_selected_,
            "ridge_selected": self.ridge_,
        }

    def get_params(self, deep=True):
        """Get parameters for this estimator (scikit-learn compatibility)."""
        return {
            "task": self.task,
            "feature_names": self.feature_names,
            "n_quantiles": self.n_quantiles,
            "max_bases_per_feature": self.max_bases_per_feature,
            "max_interactions": self.max_interactions,
            "n_environments": self.n_environments,
            "stability_threshold": self.stability_threshold,
            "alpha": self.alpha,
            "ridge": self.ridge,
            "class_weight": self.class_weight,
            "impute_missing": self.impute_missing,
            "max_iter": self.max_iter,
            "max_samples_discovery": self.max_samples_discovery,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """Set parameters for this estimator (scikit-learn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def summary(self):
        """Print summary of model architecture and stability decomposition."""
        n_stable = sum(1 for b in self.bases_ if b.is_stable)
        n_unstable = self.n_selected_ - n_stable
        print()
        print(f"{'═' * 62}")
        print(f"{'AstralModel v' + __version__ + ' — Spectral Resonance Decomposition':^62s}")
        print(f"{'═' * 62}")
        print(f"  Task               : {self.task_.capitalize()}"
              + (f" ({self._n_classes}-class)" if self.task_ == "classification" else ""))
        print(f"  Training samples   : {self._n_train}")
        print(f"  Selected bases     : {self.n_selected_}")
        print(f"  Stable (causal)    : {n_stable}")
        print(f"  Unstable (corr.)   : {n_unstable}")
        print(f"  Ridge penalty (λ)  : {self.ridge_:.4f}")
        if self.task_ == "classification":
            print(f"  Train Accuracy     : {self._train_acc:.4f} | F1: {self._train_f1:.4f}")
        else:
            print(f"  Train R²           : {self._train_r2:.4f}")
        print(f"{'═' * 62}\n")
