# ASTRAL: Adaptive Spectral Resonance Decomposition & Invariant Risk Minimization

<div align="center">

[![PyPI Version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg?style=flat-square&logo=pypi)](https://pypi.org/project/astral-model/)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg?style=flat-square&logo=python)](https://pypi.org/project/astral-model/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg?style=flat-square)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg?style=flat-square)](test_astral.py)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

**A causal-invariant, highly interpretable machine learning algorithm for tabular data.**  
*Bridging the gap between the predictive power of gradient-boosted trees and the mathematical transparency of generalized additive models.*

[Key Innovations](#-key-innovations) •
[Benchmark Comparison](#-empirical-benchmarks--algorithm-comparison) •
[Installation](#-installation) •
[Quickstart](#-quickstart) •
[API Reference](#-complete-api--function-reference) •
[Tutorials](#-in-depth-usage-tutorials) •
[Citation](#-citation)

</div>

---

## 🌟 Overview

Tabular machine learning has traditionally forced practitioners to choose between two extremes:
1. **Opaque Black Boxes:** Gradient-Boosted Decision Trees (GBDTs like XGBoost, LightGBM, CatBoost) and deep neural networks (TabNet) offer strong predictive accuracy but produce thousands of discontinuous step functions prone to shortcut learning, uncalibrated overconfidence, and lack of mathematical interpretability.
2. **Rigid Transparent Models:** Linear and logistic regressions provide full interpretability and fast inference, but fail on complex non-linear manifolds and cross-feature interactions.

**ASTRAL** (*Adaptive Spectral Resonance Decomposition & Invariant Risk Minimization*) resolves this dilemma. It operates in a continuous Reproducing Kernel Hilbert Space (RKHS), dynamically synthesises localized non-linear basis functionals, discovers pairwise non-linear synergy (resonance), penalizes spurious correlations across synthetic perturbation environments via Invariant Risk Minimization (IRM), and provides distribution-free conformal uncertainty guarantees.

Implemented in **pure NumPy** with zero mandatory C++ or heavy deep-learning dependencies, ASTRAL trains in seconds while delivering accuracy competitive with top gradient-boosted ensembles.

---

## 🔬 Key Innovations

```text
[Input Matrix X (N x d), Target y]
                 │
                 ├── 1. Univariate Adaptive Basis Expansion (Quantile Hats, Gaussian RBFs, Sigmoids, Roots, Steps)
                 │
                 ├── 2. Non-Linear Pairwise Resonance Screening (Multiplicative, Soft Ratios, Min, AbsDiff)
                 │
                 ├── 3. Invariant Risk Minimization (IRM) Stability Diagnostics (E=10 bootstrap environments)
                 │
                 ├── 4. Minimal Description Length (MDL) Forward Orthogonal Pursuit (Gram-Schmidt projection)
                 │
                 ├── 5. Stability-Weighted Convex Optimization (R_jj = λ / (0.25 + 0.75 s_j))
                 │
                 └── 6. Finite-Sample Conformal Calibration (Prediction sets, confidence intervals, abstention)
```

1. **Adaptive Spectral Resonance Expansion:** Instead of exponential polynomial or Fourier dictionaries, ASTRAL positions empirical quantile kernels (Quantile Hat B-splines, Gaussian RBFs, and Sigmoids) directly over localized data densities.
2. **Synergistic Resonance Discovery:** Evaluates pairwise interactions (multiplicative, soft ratio, minimum, and absolute differential) and retains only those exhibiting positive *Mutual Information Resonance Gain* over individual marginals:
   $$\text{Gain}(\psi; y) = I(\psi; y) - \max(I(\phi_i; y), I(\phi_j; y)) > \tau_{\text{resonance}}$$
3. **Causal Invariance Regularization:** Evaluates correlation magnitude and directionality across $E=10$ synthetic perturbation environments to compute an invariance score $s_j \in [0, 1]$. Invariant features receive minimal shrinkage, while volatile shortcut features are heavily penalized:
   $$R_{jj} = \frac{\lambda}{0.25 + 0.75 s_j}$$
4. **Greedy Orthogonal Residual Pursuit (MDL):** Modified Gram-Schmidt orthogonalization eliminates collinearity and redundant bases, balancing predictive correlation gain against basis complexity.
5. **Distribution-Free Conformal Uncertainty:** Calculates non-asymptotic $(1 - \alpha)$ prediction intervals for regression and prediction sets for classification, equipped with automated prediction abstention for high-ambiguity or out-of-distribution inputs.
6. **Robust to Unscaled Heterogeneous Data:** Quantile anchor placements make ASTRAL naturally invariant to monotonic feature scaling, avoiding convergence failures common to Neural Networks (MLPs) and Support Vector Machines.

---

## 📊 Empirical Benchmarks & Algorithm Comparison

ASTRAL was benchmarked against **10 standard machine learning algorithms** across three heterogeneous public tabular datasets:

### 1. Regression: Ames House Prices (`data-1`)
*1,460 samples, 79 continuous & categorical features. Evaluated on $R^2$ (higher is better), RMSE, and MAE (lower is better).*

| Algorithm | Model Family | Raw $R^2$ | Scaled $R^2$ | Scaled RMSE | Scaled MAE | Fit Time (s) | Interpretability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ASTRAL** | **Adaptive Resonance / IRM** | **0.8425** | **0.8596** | **32,819** | **18,413** | 1.24s | **High (Symbolic)** |
| **Gradient Boosting** | GBDT Ensemble | 0.9035 | 0.9031 | 27,257 | 17,110 | 0.54s | Low (Black-box) |
| **Random Forest** | Bagged Trees | 0.8832 | 0.8830 | 29,960 | 18,195 | 0.16s | Low (Black-box) |
| **HistGradientBoosting**| GBDT Ensemble | 0.8820 | 0.8820 | 30,089 | 17,332 | 0.38s | Low (Black-box) |
| **ElasticNet** | Regularized Linear | 0.8340 | 0.8263 | 36,506 | 21,910 | 0.01s | High (Linear only) |
| **Ridge Regression** | Regularized Linear | 0.8212 | 0.8070 | 38,475 | 22,421 | 0.01s | High (Linear only) |
| **Lasso Regression** | Regularized Linear | 0.8060 | 0.8060 | 38,575 | 22,442 | 0.01s | High (Linear only) |
| **Decision Tree** | Single Tree | 0.7931 | 0.7933 | 39,814 | 26,505 | 0.03s | Medium (Stepwise) |
| **K-Nearest Neighbors**| Instance-based | 0.7036 | 0.7824 | 40,857 | 24,747 | 0.01s | Low (Non-parametric) |
| **LinearSVR** | Support Vector Machine | 0.7171 | -4.1155 | 198,084 | 177,671 | 0.01s | Low (Diverged) |
| **MLP Regressor** | Deep Neural Network | 0.1725 | -4.1580 | 198,905 | 178,663 | 0.22s | Low (Diverged) |

### 2. Binary Classification: Electric Vehicle (EV) Adoption (`data-2`)
*15,000 samples, 13 demographic, geographical & economic features.*

| Algorithm | Model Family | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ASTRAL** | **Adaptive Resonance / IRM** | **0.8940** | **0.8139** | **0.8935** | **0.9394** | 2.40s |
| **AdaBoost** | Boosted Trees | 0.8943 | 0.8065 | 0.8915 | 0.9373 | 0.44s |
| **Gradient Boosting** | GBDT Ensemble | 0.8917 | 0.8123 | 0.8919 | 0.9376 | 1.27s |
| **HistGradientBoosting**| GBDT Ensemble | 0.8927 | 0.8141 | 0.8929 | 0.9371 | 0.36s |
| **Random Forest** | Bagged Trees | 0.8967 | 0.8110 | 0.8940 | 0.9336 | 0.22s |
| **Gaussian Naive Bayes**| Generative Probabilistic | 0.8710 | 0.7308 | 0.8575 | 0.9041 | 0.01s |
| **Decision Tree** | Single Tree | 0.8830 | 0.8025 | 0.8847 | 0.8794 | 0.04s |
| **Logistic Regression**| Linear Model (Raw) | 0.8463 | 0.7059 | 0.8384 | 0.8725 | 0.38s |
| **LinearSVC** | Support Vector Machine (Raw)| 0.8260 | 0.4524 | 0.7473 | 0.3349 | 0.01s |
| **MLP Classifier** | Deep Neural Network (Raw) | 0.8260 | 0.4524 | 0.7473 | 0.3922 | 0.61s |

### 3. Medical Classification: Cleveland Heart Disease (`data-3`)
*303 clinical samples, 13 clinical biomarkers.*

| Algorithm | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ASTRAL** | **0.8689** | **0.8665** | **0.8683** | **0.9044** | 0.34s |
| **AdaBoost** | 0.8361 | 0.8324 | 0.8344 | 0.8853 | 0.07s |
| **Random Forest** | 0.8525 | 0.8517 | 0.8523 | 0.8936 | 0.12s |
| **Gradient Boosting** | 0.8033 | 0.8024 | 0.8030 | 0.8821 | 0.14s |
| **MLP Classifier** | 0.7705 | 0.7699 | 0.7709 | 0.8258 | 0.08s |
| **Decision Tree** | 0.6885 | 0.6855 | 0.6880 | 0.6851 | 0.01s |

> **Key Takeaway:** ASTRAL outperforms or matches ensemble methods (Random Forest, AdaBoost, Gradient Boosting) on classification AUC while maintaining **exact symbolic form**, **causal invariance guarantees**, and **complete robustness to unscaled raw features**.

---

## 📦 Installation

### Standard Installation
```bash
pip install astral-model
```

### Installation with Full Optional Ecosystem (Torch, Scikit-Learn, Plotting)
```bash
pip install "astral-model[all]"
```

### From Local Source Repository
```bash
git clone https://github.com/Aditya7129p/astral-model.git
cd astral-model
pip install -e .
```

---

## ⚡ Quickstart

### 1. Classification with Automatic Regularization
```python
from astral import AstralModel, astral_train_test_split
from sklearn.datasets import load_breast_cancer

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = astral_train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Instantiate and fit ASTRAL
model = AstralModel(task="classification", ridge="auto", random_state=42)
model.fit(X_train, y_train)

# Predict labels and class probabilities
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)

# Evaluate
metrics = model.score(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
```

### 2. High-Stakes Regression with Conformal Uncertainty
```python
from astral import AstralModel, astral_train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

# Load data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = astral_train_test_split(X[:3000], y[:3000], test_size=0.2, random_state=42)

# Fit regression model
model = AstralModel(task="regression", ridge="auto", random_state=42)
model.fit(X_train, y_train)

# Predict with 95% guaranteed conformal prediction intervals (alpha=0.05)
y_pred, y_lower, y_upper = model.predict_with_uncertainty(X_test, alpha=0.05)

coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
print(f"Test R²: {model.score(X_test, y_test)['r2']:.4f}")
print(f"Empirical Conformal Coverage: {coverage * 100:.2f}% (Target: 95.00%)")
```

---

## 📖 Complete API & Function Reference

### 1. `AstralModel` Class

```python
AstralModel(
    X=None,
    y=None,
    task='auto',
    feature_names=None,
    n_quantiles=5,
    max_bases_per_feature=6,
    max_interactions=35,
    n_environments=10,
    stability_threshold=0.5,
    alpha=0.1,
    ridge='auto',
    ridge_grid=None,
    class_weight=None,
    impute_missing=True,
    max_iter=80,
    max_samples_discovery=4000,
    random_state=42,
    verbose=False
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `X` | `np.ndarray \| pd.DataFrame \| None` | `None` | Optional initial training feature matrix for one-shot construction fitting. |
| `y` | `np.ndarray \| pd.Series \| None` | `None` | Optional initial training target vector for one-shot construction fitting. |
| `task` | `str` | `'auto'` | Target problem type: `'auto'`, `'classification'`, or `'regression'`. Automatically inferred if target consists of continuous vs discrete integers. |
| `feature_names` | `list[str] \| None` | `None` | Custom names for input features. Inferred automatically if `X` is a pandas DataFrame. |
| `n_quantiles` | `int` | `5` | Number of empirical quantile anchor points per feature for placing hat, RBF, and threshold basis functions. |
| `max_bases_per_feature` | `int` | `6` | Upper limit on univariate non-linear basis transforms generated per marginal feature during Phase 2. |
| `max_interactions` | `int` | `35` | Maximum number of pairwise non-linear interactions retained in design matrix. |
| `n_environments` | `int` | `10` | Number of synthetic bootstrap perturbation environments generated for Invariant Risk Minimization (IRM) stability diagnostics. |
| `stability_threshold` | `float` | `0.5` | Threshold for IRM stability score $s_j \in [0, 1]$ to mark a feature as structurally invariant. |
| `alpha` | `float` | `0.1` | Significance level for conformal calibration (default `0.1` provides 90% finite-sample coverage). |
| `ridge` | `float \| str` | `'auto'` | L2 shrinkage penalty $\lambda$. If `'auto'`, selected via internal stratified K-fold cross-validation. |
| `ridge_grid` | `list[float] \| None` | `None` | Candidate values of $\lambda$ evaluated during cross-validation. Defaults to `[1e-4, 1e-3, ..., 1e3]`. |
| `class_weight` | `str \| dict \| None` | `None` | Class weighting scheme (`'balanced'` or custom dictionary `{0: w0, 1: w1}`) for imbalanced classification. |
| `impute_missing` | `bool` | `True` | Automatically replaces NaNs and infinities with median statistics computed during training. |
| `max_iter` | `int` | `80` | Maximum solver iterations for logistic regression optimization (Newton-CG). |
| `max_samples_discovery`| `int` | `4000` | Maximum sub-sample size used for basis synthesis and IRM ranking on large datasets ($N > 25,000$). |
| `random_state` | `int` | `42` | Random seed for environment bootstrap resampling and CV fold generation. |
| `verbose` | `bool` | `False` | If True, logs discovery, IRM stability filtering, and solver convergence details. |

#### Methods

- **`fit(X, y)`**  
  Fits the complete 6-phase ASTRAL pipeline on input matrix `X` and targets `y`. Returns `self`.
- **`predict(X)`**  
  Predicts discrete class labels for classification or continuous values for regression.  
  *Returns:* `np.ndarray` of shape `(N,)`.
- **`predict_proba(X)`**  
  Computes calibrated class probability distributions.  
  *Returns:* `np.ndarray` of shape `(N, C)` where $C$ is number of classes.
- **`predict_set(X, alpha=None, allow_abstention=True)`**  
  Outputs conformal prediction sets with guaranteed $1 - \alpha$ coverage. If a sample is too ambiguous or out-of-distribution, triggers prediction abstention.  
  *Returns:* `list[list[int]]` of prediction sets per sample.
- **`predict_with_uncertainty(X, alpha=None)`**  
  Generates point predictions alongside lower and upper conformal bound intervals.  
  *Returns:* Tuple `(y_pred, y_lower, y_upper)`.
- **`score(X, y)`**  
  Computes task-specific metrics. Returns a dictionary:
  - Classification: `{'accuracy': ..., 'f1_macro': ..., 'f1_weighted': ..., 'roc_auc': ...}`
  - Regression: `{'r2': ..., 'rmse': ..., 'mae': ...}`
- **`get_feature_importance()`**  
  Extracts ranked basis importance scores alongside IRM stability flags (`is_stable`).  
  *Returns:* `list[dict]` sorted by descending absolute importance.
- **`get_stable_features()`**  
  Returns names of all features verified to be structurally invariant across bootstrap environments.  
  *Returns:* `list[str]`.
- **`get_unstable_features()`**  
  Returns names of features exhibiting spurious or volatile behavior across environments.  
  *Returns:* `list[str]`.
- **`explain(x, top_k=5)`**  
  Decomposes an individual instance prediction into linear contributions from top discovered non-linear basis functionals.  
  *Returns:* `dict` with `'prediction'`, `'intercept'`, and `'contributions'`.
- **`summary()`**  
  Prints a human-readable table of active bases, mathematical transforms, weights, and IRM stability.
- **`data_quality_report()`**  
  Inspects missing rates, infinite values, constant columns, and cardinality statistics.  
  *Returns:* `dict`.
- **`get_params(deep=True)` / `set_params(**params)`**  
  Full Scikit-Learn estimator parameter inspection and mutation compatibility.

---

### 2. Utility Classes & Functions

#### `AstralScaler()`
Drop-in replacement for scikit-learn's `StandardScaler`, implemented in pure NumPy.
```python
from astral import AstralScaler

scaler = AstralScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_orig = scaler.inverse_transform(X_test_scaled)
```

#### `astral_train_test_split(X, y, test_size=0.2, random_state=None, stratify=None)`
Fast, pure-NumPy stratified train/test partitioning.
```python
from astral import astral_train_test_split

X_train, X_test, y_train, y_test = astral_train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

#### `astral_cross_validate(X, y, task='auto', folds=5, random_state=42, **model_kwargs)`
K-Fold cross-validation harness returning average accuracy, F1, and R² scores.
```python
from astral import astral_cross_validate

results = astral_cross_validate(X, y, folds=5, random_state=42)
print("Mean F1:", results["mean_f1_macro"])
```

#### `ResonanceBasis(feature_indices, transform_type, params=None, name=None)`
Base class for all functional basis representations in ASTRAL.
- `evaluate(X)`: Evaluates transform on input matrix.
- `description_length()`: Returns MDL complexity penalty.

---

## 🛠️ In-Depth Usage Tutorials

### Tutorial 1: Inspecting Discovered Basis Functions & Invariance
```python
from astral import AstralModel
import pandas as pd

# Load Heart disease dataset
df = pd.read_csv("data-3/Heart.csv")
X = df.drop(columns=["AHD"]).select_dtypes(include=["number"]).values
y = (df["AHD"] == "Yes").astype(int).values

model = AstralModel(task="classification", ridge="auto", random_state=42)
model.fit(X, y)

# 1. Inspect Top Bases
print("Top Discovered Mathematical Bases:")
for b in model.get_feature_importance()[:5]:
    status = "✓ INVARIANT" if b["is_stable"] else "✗ UNSTABLE"
    print(f"  {status:<14} {b['name']:<30} Weight: {b['weight']:+.4f} (Importance: {b['importance']:.4f})")

# 2. Local Instance Explanation
expl = model.explain(X[0], top_k=3)
print(f"\nInstance 0 Prediction: {expl['prediction']:.4f} (Base Intercept: {expl['intercept']:+.4f})")
for c in expl["contributions"]:
    print(f"  Contribution from {c['basis_name']}: {c['value']:+.4f}")
```

### Tutorial 2: Conformal Abstention on Ambiguous Inputs
```python
# Enable prediction abstention for high-stakes decisions
pred_sets = model.predict_set(X_test, alpha=0.05, allow_abstention=True)

for i, p_set in enumerate(pred_sets[:10]):
    if len(p_set) == 0:
        print(f"Sample {i}: [ABSTAIN] Input falls outside confidence region. Deferring to human expert.")
    elif len(p_set) > 1:
        print(f"Sample {i}: [AMBIGUOUS] Candidate classes: {p_set}")
    else:
        print(f"Sample {i}: [CERTAIN] Confident class: {p_set[0]}")
```

### Tutorial 3: Model Persistence (Pickle & Joblib)
```python
import joblib

# Save fitted model
joblib.dump(model, "astral_heart_model.pkl")

# Load and predict
loaded_model = joblib.load("astral_heart_model.pkl")
preds = loaded_model.predict(X_test)
```

---

## 📂 Repository Layout

```text
astral-model/
├── astral/                     # Official Python package namespace
│   ├── __init__.py             # Public symbol exports & metadata
│   └── model.py                # Spectral Resonance & IRM engine implementation
├── docs/                       # Complete documentation suite
│   ├── figures/                # High-resolution publication figures (PNG)
│   ├── ASTRAL_DEVELOPER_GUIDE.html  # Interactive developer manual
│   ├── ASTRAL_DEVELOPER_GUIDE.md    # Markdown developer specification
│   ├── ASTRAL_USER_GUIDE.html       # Interactive user guide
│   ├── ASTRAL_USER_GUIDE.md         # Markdown user manual
│   ├── ASTRAL_RESEARCH_PAPER.html   # Full research preprint
│   ├── ASTRAL_RESEARCH_PAPER.pdf    # Publication preprint PDF
│   └── ASTRAL_RESEARCH_PAPER.tex    # Synchronized LaTeX manuscript
├── data-1/                     # Ames House Prices (Regression: 1,460 rows, 79 cols)
├── data-2/                     # EV Adoption (Binary Classification: 668k rows)
├── data-3/                     # Cleveland Heart Disease (Binary: 303 rows)
├── astral_model.py             # Standalone drop-in module
├── benchmark_models.py         # Multi-dataset 10-baseline benchmarking suite
├── benchmark_results.json      # Full empirical baseline metrics
├── test_astral.py              # Automated test suite (v0.2.0)
├── pyproject.toml              # Modern PEP 517/518 build configuration
├── setup.py                    # Backward-compatibility installer
├── LICENSE                     # Apache License 2.0
├── MANIFEST.in                 # Source distribution manifest
└── README.md                   # Repository landing page
```

---

## 📄 Citation

If you use ASTRAL in academic research or industrial projects, please cite:

```bibtex
@article{pandey2026astral,
  title   = {ASTRAL: Adaptive Spectral Resonance Decomposition and Invariant Risk Minimization for Interpretable Tabular Machine Learning},
  author  = {Aditya Pandey},
  journal = {Preprint},
  year    = {2026},
  url     = {https://github.com/Aditya7129p/astral-model}
}
```

---

## 📜 License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for complete terms.

**Author:** [Aditya Pandey](mailto:aditya9708p@gmail.com)  
Department of Computer Science & Engineering (AI & ML)  
Noida Institute of Engineering and Technology (NIET), Greater Noida, India
