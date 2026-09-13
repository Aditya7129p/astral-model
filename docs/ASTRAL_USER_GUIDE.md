# ASTRAL User Guide & Practical API Reference

> **ASTRAL: Adaptive Spectral Resonance Decomposition & Invariant Risk Minimization for Interpretable Tabular Machine Learning**  
> *Author:* Aditya Pandey (<aditya9708p@gmail.com>) &bull; *Version:* 0.2.1 &bull; *License:* Apache-2.0

---

## 1. Installation

ASTRAL is written in pure NumPy with zero mandatory external C++ or deep learning compiler dependencies:

```bash
# Install official package from PyPI
pip install astral-model

# Or install with optional extras (Torch acceleration, Scikit-Learn benchmarking)
pip install "astral-model[all]"

# Editable installation from local source repository
pip install -e .
```

---

## 2. 5-Minute Quickstart

ASTRAL implements the Scikit-Learn estimator pattern (`fit`, `predict`, `predict_proba`, `score`):

```python
from astral import AstralModel, AstralScaler, astral_train_test_split
from sklearn.datasets import load_breast_cancer

# 1. Load sample dataset
X, y = load_breast_cancer(return_X_y=True)

# 2. Train/Test split with stratification
X_train, X_test, y_train, y_test = astral_train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Fit scaler & model
scaler = AstralScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = AstralModel(ridge="auto", random_state=42)
model.fit(X_train_s, y_train)

# 4. Predict & evaluate
y_pred = model.predict(X_test_s)
metrics = model.score(X_test_s, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['f1_macro']:.4f}")
```

---

## 3. Constructor Parameters Reference

All parameters accepted by `AstralModel(...)`:

| Parameter | Type / Default | Description & Guidance |
| :--- | :--- | :--- |
| `task` | `str = 'auto'` | `'auto'`, `'classification'`, or `'regression'`. In auto mode, inferred from target discrete/continuous characteristics. |
| `n_quantiles` | `int = 5` | Number of empirical quantile anchor points per feature for hat, RBF, and threshold basis placements. |
| `max_bases_per_feature`| `int = 6` | Maximum univariate non-linear basis functionals generated per raw marginal feature. |
| `max_interactions` | `int = 35` | Maximum pairwise resonance interactions (multiplicative, ratio, min, diff) retained in design dictionary. |
| `n_environments` | `int = 10` | Number of synthetic bootstrap perturbation environments generated to evaluate Invariant Risk Minimization (IRM) stability. |
| `stability_threshold`| `float = 0.5` | Minimum IRM stability score $s_j$ required for a feature to be considered structurally invariant. |
| `alpha` | `float = 0.1` | Significance level for conformal prediction intervals and sets (default 0.1 provides 90% coverage guarantee). |
| `ridge` | `float \| str = 'auto'`| L2 regularization parameter $\lambda$. When `'auto'`, automatically tuned via stratified K-fold cross-validation. |
| `ridge_grid` | `list \| None = None` | Custom candidate regularizer values for CV grid search. Defaults to logarithmic mesh `[1e-4, ..., 1e3]`. |
| `class_weight` | `str \| dict \| None = None` | `'balanced'` or custom class-to-weight dictionary to counter severe target class imbalance. |
| `impute_missing` | `bool = True` | Automatically replaces NaNs and infinite values with column medians learned during training. |
| `max_samples_discovery`| `int = 4000` | Maximum sub-sample size used for basis synthesis and IRM ranking on large datasets ($N > 25,000$). |
| `random_state` | `int = 42` | Random seed for reproducible environment bootstrap resampling and CV splits. |
| `verbose` | `bool = False` | If True, logs progress of basis discovery, IRM stability filtering, and regularizer selection. |

---

## 4. Methods & API Reference

| Method | Signature | Returns & Functionality |
| :--- | :--- | :--- |
| `fit(X, y)` | `(X, y)` | Fits the 6-phase ASTRAL model pipeline. Returns `self`. |
| `predict(X)` | `(X) -> np.ndarray` | Predicts class labels for classification or continuous targets for regression. |
| `predict_proba(X)` | `(X) -> np.ndarray` | Calibrated class probability distributions of shape `(N, C)`. |
| `predict_set(X, alpha)` | `(X, alpha=None, allow_abstention=True) -> list[list]` | Conformal prediction sets with guaranteed $1 - \alpha$ coverage. |
| `predict_with_uncertainty(X, alpha)`| `(X, alpha=None) -> (y_pred, y_lower, y_upper)` | Point predictions and exact conformal prediction interval bounds. |
| `score(X, y)` | `(X, y) -> dict` | Computes classification (accuracy, F1, ROC-AUC) or regression ($R^2$, RMSE, MAE) metrics. |
| `get_feature_importance()`| `() -> list[dict]` | Ranked basis importance scores alongside IRM stability flags. |
| `get_stable_features()` | `() -> list[str]` | Returns names of all features verified invariant across bootstrap perturbation environments. |
| `explain(x, top_k)` | `(x, top_k=5) -> dict` | Generates local linear basis decomposition explaining an individual instance prediction. |
| `summary()` | `() -> str` | Formatted textual summary of active bases, IRM stability, and learned weights. |
| `data_quality_report()` | `() -> dict` | Inspects NaN rates, infinite values, constant columns, and imputation summary. |

---

## 5. Classification with Conformal Prediction Sets

```python
from astral import AstralModel
import numpy as np

# Train model
model = AstralModel(task="classification", ridge="auto", random_state=42)
model.fit(X_train, y_train)

# Conformal prediction sets with 90% guaranteed coverage (alpha=0.1)
pred_sets = model.predict_set(X_test, alpha=0.1)

# Inspect predictions
for i in range(5):
    probs = model.predict_proba(X_test[i:i+1])[0]
    print(f"Sample {i}: Pred={model.predict(X_test[i:i+1])[0]}, Set={pred_sets[i]}, Probs={np.round(probs, 3)}")
```

---

## 6. Regression & Uncertainty Intervals

```python
model = AstralModel(task="regression", ridge="auto", random_state=42)
model.fit(X_train, y_train)

# Predict with calibrated conformal interval
y_pred, y_lower, y_upper = model.predict_with_uncertainty(X_test, alpha=0.05) # 95% coverage

# Calculate empirical coverage
coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
print(f"Empirical Coverage: {coverage * 100:.2f}% (Target: 95%)")
print(f"Average Interval Width: {np.mean(y_upper - y_lower):.2f}")
```

---

## 7. Model Explainability & Invariance Inspection

```python
# 1. Inspect global basis importances and causal invariance
for item in model.get_feature_importance()[:8]:
    status = "[INVARIANT]" if item["is_stable"] else "[UNSTABLE]"
    print(f"{status:<12} {item['name']:<30} Importance: {item['importance']:.4f}")

# 2. Local instance explanation
explanation = model.explain(X_test[0], top_k=3)
print("\nLocal Instance Prediction Breakdown:")
for contribution in explanation["contributions"]:
    print(f"  {contribution['basis_name']}: contribution = {contribution['value']:+.4f}")
```

---

## 8. Empirical Benchmark Results

| Dataset | Task & Samples | ASTRAL | Gradient Boosting | Random Forest | Logistic / Ridge |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **data-1 (Ames Housing)** | Regression (1,460 rows, 79 cols) | **$R^2$ = 0.8596** (RMSE: 32,819) | $R^2$ = 0.9031 | $R^2$ = 0.8830 | $R^2$ = 0.8070 |
| **data-2 (EV Adoption)** | Classification (15,000 samples) | **AUC = 0.9394** (Acc: 89.4%) | AUC = 0.9376 | AUC = 0.9336 | AUC = 0.8725 (raw) |
| **data-3 (Heart Disease)** | Classification (303 samples) | **AUC = 0.9044** (Acc: 86.9%) | AUC = 0.8821 | AUC = 0.8936 | AUC = 0.8853 |

---

## 9. Frequently Asked Questions (FAQ)

- **Q: Do I need to normalize or scale features before training?**  
  *A:* While `AstralScaler` is provided for convenience, ASTRAL is intrinsically robust to unscaled heterogeneous data because its non-linear basis transforms are positioned directly at empirical training quantiles.
- **Q: How does ASTRAL handle missing values (NaNs)?**  
  *A:* By default (`impute_missing=True`), ASTRAL automatically detects missing values, computes training column medians, and fills missing inputs at both fit and predict time without data leakage.
- **Q: Can ASTRAL be pickled or saved with Joblib?**  
  *A:* Yes. ASTRAL uses pure standard Python data structures and NumPy arrays, making it 100% serializable via standard `pickle` or `joblib.dump()`.
