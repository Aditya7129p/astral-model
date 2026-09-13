"""
AstralModel - Spectral Resonance Decomposition (SRD) & Invariant Risk Minimization
===================================================================================

A causal-invariant and interpretable machine learning algorithm that unifies:
  - Adaptive spectral resonance basis functions (Quantile Hats, Gaussian RBF, Sigmoids, Roots, Powers)
  - Non-linear pairwise interaction discovery (Multiplicative, Soft Ratios, Min, AbsDiff)
  - Invariant Risk Minimization (IRM) stability diagnostics across perturbed bootstrap environments
  - Stability-weighted regularized fitting (invariant features receive lower shrinkage)
  - Fast stratified cross-validation for optimal regularizer selection
  - Calibrated uncertainty via conformal prediction intervals & sets with optional abstention
  - Minimal Description Length (MDL) greedy orthogonalized basis pruning
  - Full Scikit-Learn API compatibility (fit, predict, predict_proba, score)

Author: Aditya Pandey <aditya9708p@gmail.com>
License: Apache-2.0
"""

import sys
import os

# Support direct execution and standard package layout
try:
    from astral_model import (
        AstralModel,
        AstralScaler,
        astral_train_test_split,
        astral_cross_validate,
        ResonanceBasis,
        __version__,
        __version_info__,
        __author__,
        __license__,
        version,
    )
except ImportError:
    # If installed as isolated package without astral_model at top-level
    try:
        from .model import (
            AstralModel,
            AstralScaler,
            astral_train_test_split,
            astral_cross_validate,
            ResonanceBasis,
            __version__,
            __version_info__,
            __author__,
            __license__,
            version,
        )
    except ImportError:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from astral_model import (
            AstralModel,
            AstralScaler,
            astral_train_test_split,
            astral_cross_validate,
            ResonanceBasis,
            __version__,
            __version_info__,
            __author__,
            __license__,
            version,
        )

__all__ = [
    "AstralModel",
    "AstralScaler",
    "astral_train_test_split",
    "astral_cross_validate",
    "ResonanceBasis",
    "__version__",
    "__version_info__",
]
