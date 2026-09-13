# ASTRAL Developer Guide & System Architecture

> **ASTRAL: Adaptive Spectral Resonance Decomposition & Invariant Risk Minimization for Interpretable Tabular Machine Learning**  
> *Author:* Aditya Pandey (<aditya9708p@gmail.com>) &bull; *Version:* 0.2.1 &bull; *License:* Apache-2.0

---

## 1. Core Philosophy & Design Rationale

Tabular machine learning has long suffered an acute dichotomy between **opaque gradient-boosted decision trees (GBDTs) / deep neural networks** and **rigid linear/logistic models**:

- **The Ensembling Problem:** GBDTs (XGBoost, LightGBM, CatBoost) achieve high empirical accuracy by recursively partitioning feature space, but generate thousands of discontinuous step functions that cannot be audited, verified for causal invariant behavior, or proven free of spurious shortcut correlations.
- **The Neural Tabular Problem:** Deep tabular models (TabNet, FT-Transformer) incur immense computational training overhead, require delicate learning rate schedules, struggle with heterogeneous unscaled data, and remain black-box function approximators.
- **The ASTRAL Alternative:** ASTRAL operates in a continuous Reproducing Kernel Hilbert Space (RKHS) by dynamically synthesising empirical quantile basis functions, isolating non-linear cross-feature resonance, penalizing unstable shortcut features via Invariant Risk Minimization (IRM), and solving a closed-form convex regularized objective.

---

## 2. Mathematical Foundations

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ denote a dataset where $x_i \in \mathbb{R}^d$ and $y_i \in \mathcal{Y}$. ASTRAL models the response function $f(x)$ as a linear combination of discovered non-linear basis functionals $\phi_k: \mathbb{R}^d \to \mathbb{R}$:

$$f(x) = w_0 + \sum_{k=1}^M w_k \phi_k(x) = w_0 + w^T \Phi(x)$$

Unlike fixed polynomial or Fourier expansions whose basis size suffers from the curse of dimensionality $\mathcal{O}(d^p)$, ASTRAL discovers an adaptive, highly sparse dictionary $\mathcal{B} = \{\phi_1, \dots, \phi_M\}$ with $M \ll d^2$ via mutual information resonance screening and Minimal Description Length (MDL) pruning.

---

## 3. Basis Function Dictionary

For each marginal feature $X_j$, ASTRAL computes $Q$ empirical quantiles $q_1 < q_2 < \dots < q_Q$ and evaluates six complementary non-linear basis families:

| Basis Family | Mathematical Formulation | Structural Target |
| :--- | :--- | :--- |
| **Linear** | $\phi(x) = x_j$ | Global monotonic trends and baseline linear relationships |
| **Quantile Hat (Spline)** | $\Lambda(x; q_{k-1}, q_k, q_{k+1}) = \max(0, 1 - \|x - q_k\| / \Delta)$ | Piecewise linear local regimes and localized modal densities |
| **Gaussian RBF** | $\kappa(x; c, \sigma) = \exp\left(-\frac{(x - c)^2}{2\sigma^2}\right)$ | Smooth bell-shaped clustering and non-linear localized attraction |
| **Sigmoidal Transition** | $\sigma(x; c, w) = \frac{1}{1 + \exp(-w(x - c))}$ | Smooth binary phase shifts and threshold saturation phenomena |
| **Root & Power** | $\phi(x) = \text{sign}(x)\sqrt{\|x\|},\; \phi(x) = x^2$ | Sub-linear diminishing returns and super-linear quadratic acceleration |
| **Heaviside Step** | $\theta(x; q_k) = \mathbb{I}(x \ge q_k)$ | Hard administrative boundaries, policy thresholds, and categorical splits |

---

## 4. Pairwise Non-Linear Resonance Screening

Given top informative marginal bases $\phi_i$ and $\phi_j$, ASTRAL constructs candidate bivariate interaction functionals:

$$\psi_{\text{mult}} = \phi_i(x) \cdot \phi_j(x), \quad \psi_{\text{ratio}} = \frac{\phi_i(x)}{|\phi_j(x)| + \epsilon}, \quad \psi_{\text{min}} = \min(\phi_i(x), \phi_j(x)), \quad \psi_{\text{diff}} = |\phi_i(x) - \phi_j(x)|$$

A candidate interaction functional $\psi$ is admitted into the dictionary if and only if it exhibits **Resonance Gain** over both individual constituents:

$$\text{Gain}(\psi; y) = I(\psi; y) - \max\left(I(\phi_i; y), I(\phi_j; y)\right) > \tau_{\text{resonance}}$$

This guarantees that the model only incorporates interactions that capture genuine synergistic non-linear information, preventing combinatorial explosion.

---

## 5. Invariant Risk Minimization (IRM) Stability Diagnostics

Standard empirical risk minimization (ERM) eagerly exploits spurious correlations that happen to align with targets in the training sample. ASTRAL mitigates this by generating $E$ synthetic perturbation environments $\mathcal{E} = \{e_1, e_2, \dots, e_E\}$ through bootstrap resampling and adversarial feature noise injection.

For each candidate basis functional $\phi_k$, ASTRAL computes the correlation vector across all environments $\rho_k = [\text{corr}^{(e)}(\phi_k, y)]_{e \in \mathcal{E}}$ and calculates an invariance stability score $s_k \in [0, 1]$:

$$s_k = \left| \mu(\rho_k) \right| \cdot \left[ 1 - \frac{\sigma(\rho_k)}{\epsilon + |\mu(\rho_k)|} \right]_+$$

- **Invariance Rule:** If a basis functional displays erratic correlation flips across resampled splits, its stability score $s_k \to 0$. If it reliably maintains consistent directionality and high signal-to-noise ratio across all splits, $s_k \to 1$.

---

## 6. Minimal Description Length & Gram-Schmidt Selection

To prevent collinear redundancy and over-parameterization, candidate bases are selected sequentially using greedy forward orthogonal residual pursuit:

1. Initialize residual vector $r_0 = y - \bar{y}$ and active basis set $\mathcal{S}_0 = \emptyset$.
2. At step $m$, project each candidate basis $\phi_j$ onto the orthogonal complement of the subspace spanned by $\mathcal{S}_{m-1}$ using modified Gram-Schmidt:
   $$\tilde{\phi}_j^{(m)} = \phi_j - \sum_{k \in \mathcal{S}_{m-1}} \frac{\langle \phi_j, u_k \rangle}{\|u_k\|^2} u_k$$
3. Score candidate $\phi_j$ by trading off residual correlation against stability and Description Length penalty:
   $$\mathcal{J}(\phi_j) = \frac{|\langle r_{m-1}, \tilde{\phi}_j^{(m)} \rangle|}{\|\tilde{\phi}_j^{(m)}\|} \cdot \left(0.5 + 0.5 s_j\right) - \gamma \cdot \text{MDL}(\phi_j)$$
4. Add the maximizer $\phi^* = \arg\max_j \mathcal{J}(\phi_j)$ to $\mathcal{S}_m$ and update residual $r_m$. Terminate when gain drops below threshold or max bases budget is exhausted.

---

## 7. Stability-Weighted Regularized Fitting

Once the active basis design matrix $\mathbf{\Phi} \in \mathbb{R}^{N \times M}$ is assembled, the model solves the convex stability-weighted regularized problem:

$$\min_{w} \; \mathcal{L}(y, \mathbf{\Phi} w) + \frac{1}{2} w^T \mathbf{R} w$$

where $\mathbf{R}$ is a diagonal regularization matrix whose penalty dynamically scales inversely with causal stability:

$$R_{jj} = \frac{\lambda}{0.25 + 0.75 s_j}$$

- **Regression:** Closed-form normal equations:
  $$w^* = \left( \mathbf{\Phi}^T \mathbf{\Phi} + \mathbf{R} \right)^{-1} \mathbf{\Phi}^T y$$
- **Classification:** Solved via Newton-CG optimization with analytical gradient and Hessian vector products.

---

## 8. Conformal Uncertainty & Abstention Engine

ASTRAL incorporates non-asymptotic distribution-free conformal calibration. Let $\alpha \in (0, 1)$ be the user error rate budget:

- **Conformal Prediction Intervals (Regression):** Computes absolute residual nonconformity scores $R_i = |y_i - \hat{y}_i|$. Finds the empirical $(1-\alpha)(1 + 1/N)$ quantile $\hat{q}$. Guarantees finite-sample coverage:
  $$P\left(Y \in [f(x) - \hat{q}, f(x) + \hat{q}]\right) \ge 1 - \alpha$$
- **Conformal Prediction Sets (Classification):** Computes cumulative softmax nonconformity. Returns guaranteed prediction sets $C(x) \subseteq \mathcal{Y}$. If $|C(x)| > 1$ or predicted class probability falls below the confidence margin, triggers automated prediction abstention.

---

## 9. Six-Phase Execution Pipeline

```text
[Input Data Matrix X (N x d), Target y]
                     │
    Phase 1 ─────────▼───────── Input Validation & Imputation
                     │          • Median imputation for NaNs / infinities
                     │          • Empirical quantile mesh construction
    Phase 2 ─────────▼───────── Univariate Basis Synthesis
                     │          • Linear, Quantile Hats, RBFs, Sigmoids, Roots, Steps
                     │          • Mutual information marginal filtering
    Phase 3 ─────────▼───────── Non-Linear Resonance Interaction
                     │          • Multiplicative, Ratio, Min, Difference combinations
                     │          • Positive resonance synergy screening: I(a,b; y) > max(I_a, I_b)
    Phase 4 ─────────▼───────── Perturbation Environments & IRM
                     │          • E=10 bootstrapped environments
                     │          • Stability score computation: s_j in [0, 1]
    Phase 5 ─────────▼───────── Orthogonal Gram-Schmidt & MDL Pruning
                     │          • Residual projection and collinearity elimination
                     │          • Sparse dictionary selection (M << d^2)
    Phase 6 ─────────▼───────── Stability-Weighted Optimization
                                • R_jj = lambda / (0.25 + 0.75 s_j)
                                • Direct normal equations (regression) or Newton-CG (classification)
                                • Calibrated conformal intervals & prediction sets
```

---

## 10. Performance & Scalability

- **Subsampled Discovery:** For large tabular datasets ($N > 25,000$), ASTRAL executes basis screening and stability diagnostics on a stratified sub-sample ($N_{\text{sub}} = 4,000$ default, configurable). This slashes discovery latency from minutes to **under 2.5 seconds** on datasets with 600,000+ rows.
- **Zero C++ / Pure NumPy:** The entire pipeline is vectorized with contiguous memory allocations, SIMD-friendly linear algebra, and zero external binary compiled dependencies.

---

## 11. Developer Extension Points

Developers can easily inject custom basis transforms or regularizers by extending the `ResonanceBasis` interface:

```python
from astral import ResonanceBasis
import numpy as np

class CustomPeriodicBasis(ResonanceBasis):
    """Custom Fourier / Periodic resonance basis."""
    def __init__(self, feature_idx, frequency, name=None):
        super().__init__(feature_indices=[feature_idx], transform_type="periodic", name=name)
        self.frequency = frequency

    def evaluate(self, X):
        col = X[:, self.feature_indices[0]]
        return np.sin(self.frequency * col)

    def description_length(self):
        return 2.5  # Slightly higher complexity penalty for periodic functions
```

---

## 12. Verification & Testing

```bash
# Run complete test suite
python test_astral.py

# Run multi-dataset benchmark comparison against 10 baselines
python benchmark_models.py
```
