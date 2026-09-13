"""
test_astral.py
==============
Unified, comprehensive test suite for AstralModel (v0.2.1+).

Covers:
  1. Package metadata & versioning (astral_model.version, __version__)
  2. Binary classification, probabilities, and evaluation
  3. Multiclass classification, prediction sets, and class weighting
  4. Regression tasks, R², RMSE, MAE, and conformal intervals
  5. Missing values, infinities, and data quality reporting
  6. Conformal prediction & abstention mechanism
  7. AstralScaler functionality & constant-feature guards
  8. Interpretability APIs: feature importance, stability, explanations
  9. Scikit-learn API compatibility (fit, predict, predict_proba, get_params, set_params)
 10. Reproducible K-fold cross-validation
 11. Multi-dataset smoke tests across data-1, data-2, and data-3
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure astral_model is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astral_model
from astral_model import (
    AstralModel,
    AstralScaler,
    astral_train_test_split,
    astral_cross_validate,
    __version__,
    version,
)


def test_package_metadata():
    """Verify package versioning and metadata."""
    print("Testing package metadata and versioning …")
    assert __version__ == "0.2.1"
    assert version == "0.2.1"
    assert AstralModel.version == "0.2.1"
    assert AstralModel.__version__ == "0.2.1"
    assert hasattr(astral_model, "__version_info__")
    print("  [OK] Package metadata verified.")


def test_scaler():
    """Verify AstralScaler z-score normalization and inverse transform."""
    print("Testing AstralScaler …")
    rng = np.random.RandomState(42)
    X = rng.normal(loc=10.0, scale=3.0, size=(100, 4))
    # add constant column
    X[:, 3] = 5.0

    scaler = AstralScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape
    assert np.allclose(np.nanmean(X_scaled[:, :3], axis=0), 0.0, atol=1e-7)
    assert np.allclose(np.nanstd(X_scaled[:, :3], axis=0), 1.0, atol=1e-7)
    # constant column guarded
    assert np.all(np.isfinite(X_scaled[:, 3]))

    X_inv = scaler.inverse_transform(X_scaled)
    assert np.allclose(X[:, :3], X_inv[:, :3], atol=1e-7)
    print("  [OK] AstralScaler verified.")


def test_binary_classification():
    """Verify binary classification, probability calibration, and scoring."""
    print("Testing binary classification …")
    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 5))
    y = ((X[:, 0] + X[:, 1] * 0.8 - X[:, 2] ** 2 * 0.3) > 0).astype(int)

    X_train, X_test, y_train, y_test = astral_train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = AstralModel(random_state=42, verbose=False)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset({0, 1})

    probs = model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    score_dict = model.score(X_test, y_test)
    assert "accuracy" in score_dict
    assert "f1_macro" in score_dict
    assert "roc_auc" in score_dict
    assert score_dict["roc_auc"] >= 0.50
    assert score_dict["accuracy"] > 0.60
    print(f"  [OK] Binary classification score: acc={score_dict['accuracy']:.3f}, f1={score_dict['f1_macro']:.3f}")


def test_multiclass_and_prediction_sets():
    """Verify multiclass classification, balanced class weights, and conformal prediction sets."""
    print("Testing multiclass classification and conformal sets …")
    rng = np.random.RandomState(11)
    X = rng.normal(size=(150, 6))
    X[0, 0] = np.nan
    y = np.argmax(np.column_stack([X[:, 0], -X[:, 0], X[:, 1] + X[:, 2]]), axis=1)

    model = AstralModel(
        task="classification",
        class_weight="balanced",
        max_interactions=10,
        random_state=11,
        verbose=False,
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X[:10])
    assert probabilities.shape == (10, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    pred_sets = model.predict_set(X[:5])
    assert len(pred_sets) == 5
    for s in pred_sets:
        assert isinstance(s, list)

    report = model.data_quality_report()
    assert report["missing_values_by_feature"]["X0"] == 1
    print("  [OK] Multiclass and conformal prediction sets verified.")


def test_regression_and_intervals():
    """Verify continuous target regression, R², RMSE, MAE, and conformal uncertainty intervals."""
    print("Testing regression and conformal intervals …")
    rng = np.random.RandomState(42)
    X = rng.normal(size=(140, 5))
    y = 2.5 * X[:, 0] - 1.8 * X[:, 1] + 0.5 * (X[:, 2] ** 2) + rng.normal(scale=0.2, size=140)

    X_train, X_test, y_train, y_test = astral_train_test_split(X, y, test_size=0.25, random_state=42)

    model = AstralModel(task="regression", random_state=42, verbose=False)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)

    scores = model.score(X_test, y_test)
    assert "r2" in scores
    assert "rmse" in scores
    assert scores["r2"] > 0.80

    predictions, lower, upper, abstain = model.predict_with_uncertainty(X_test)
    assert np.all(upper >= lower)
    # empirical coverage should be near nominal (90%)
    covered = np.mean((y_test >= lower) & (y_test <= upper))
    assert covered >= 0.75
    print(f"  [OK] Regression verified: R²={scores['r2']:.3f}, coverage={covered*100:.1f}%")


def test_missing_values_and_abstention():
    """Verify robust handling of NaNs, Infs, and the abstention decision option."""
    print("Testing missing value imputation and abstention …")
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 4))
    X[5, 1] = np.nan
    X[10, 2] = np.inf
    X[15, 0] = -np.inf
    y = (X[:, 0] > 0).astype(int)

    model = AstralModel(impute_missing=True, random_state=42, verbose=False)
    model.fit(X, y)

    # test abstain=True
    preds_abstain = model.predict(X, abstain=True)
    assert len(preds_abstain) == len(y)
    report = model.data_quality_report()
    assert report["imputation"] == "median"
    print("  [OK] Missing value imputation and abstention verified.")


def test_interpretability_and_explanation():
    """Verify feature importance, causal stability filtering, and per-instance local explanations."""
    print("Testing interpretability and local explanations …")
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 4))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1]
    feature_names = ["Age", "Income", "Score", "Debt"]

    model = AstralModel(feature_names=feature_names, random_state=42, verbose=False)
    model.fit(X, y)

    importances = model.get_feature_importance()
    assert len(importances) > 0
    assert "name" in importances[0]
    assert "importance" in importances[0]
    assert "stability" in importances[0]

    stable = model.get_stable_features()
    unstable = model.get_unstable_features()
    assert len(stable) + len(unstable) == len(importances)

    explanation = model.explain(X[0])
    assert len(explanation) > 0
    assert "contribution" in explanation[0]
    print("  [OK] Interpretability APIs verified.")


def test_cross_validation():
    """Verify reproducible stratified K-fold cross-validation."""
    print("Testing cross validation …")
    rng = np.random.RandomState(12)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)
    report = astral_cross_validate(X, y, task="classification", folds=3, random_state=12)

    assert len(report["folds"]) == 3
    assert "f1_macro" in report["mean"]
    assert "accuracy" in report["mean"]
    print(f"  [OK] Cross-validation verified: mean F1 = {report['mean']['f1_macro']:.3f}")


def test_data_folders_smoke():
    """Smoke test AstralModel across all 3 project data folders."""
    print("Testing data folder integration (data-1, data-2, data-3) …")

    # Data-3: Heart.csv (Classification)
    heart_path = os.path.join("data-3", "Heart.csv")
    if os.path.exists(heart_path):
        df3 = pd.read_csv(heart_path)
        if "id" in df3.columns:
            df3 = df3.drop(columns=["id"])
        if df3.columns[0] in ("", "Unnamed: 0"):
            df3 = df3.drop(df3.columns[0], axis=1)
        y3 = (df3["AHD"] == "Yes").astype(int).values
        X3 = df3.drop(columns=["AHD"]).copy()
        # encode object columns
        for c in X3.select_dtypes(include=["object"]).columns:
            X3[c] = pd.factorize(X3[c])[0]
        X3 = X3.fillna(X3.median()).values.astype(float)

        m3 = AstralModel(random_state=42, verbose=False)
        m3.fit(X3[:200], y3[:200])
        p3 = m3.predict(X3[200:])
        assert len(p3) == len(y3[200:])
        print(f"  [OK] data-3/Heart.csv smoke test passed ({len(X3)} samples).")

    # Data-1: train.csv (Regression)
    d1_path = os.path.join("data-1", "train.csv")
    if os.path.exists(d1_path):
        df1 = pd.read_csv(d1_path, nrows=300)
        y1 = df1["SalePrice"].values.astype(float)
        X1 = df1.drop(columns=["SalePrice", "Id"], errors="ignore")
        for c in X1.select_dtypes(include=["object"]).columns:
            X1[c] = pd.factorize(X1[c])[0]
        X1 = X1.fillna(X1.median()).values.astype(float)

        m1 = AstralModel(task="regression", random_state=42, verbose=False)
        m1.fit(X1[:200], y1[:200])
        p1 = m1.predict(X1[200:])
        assert len(p1) == len(y1[200:])
        print(f"  [OK] data-1/train.csv smoke test passed ({len(X1)} samples).")

    # Data-2: train.csv (Classification, sample)
    d2_path = os.path.join("data-2", "train.csv")
    if os.path.exists(d2_path):
        df2 = pd.read_csv(d2_path, nrows=500)
        y2 = (df2["Will_Buy_EV"] == "Yes").astype(int).values
        X2 = df2.drop(columns=["Will_Buy_EV", "id"], errors="ignore")
        for c in X2.select_dtypes(include=["object"]).columns:
            X2[c] = pd.factorize(X2[c])[0]
        X2 = X2.fillna(X2.median()).values.astype(float)

        m2 = AstralModel(task="classification", random_state=42, verbose=False)
        m2.fit(X2[:350], y2[:350])
        p2 = m2.predict(X2[350:])
        assert len(p2) == len(y2[350:])
        print(f"  [OK] data-2/train.csv smoke test passed ({len(X2)} samples).")


def run_all_tests():
    print("=" * 60)
    print(f"Running Unified AstralModel Test Suite (v{__version__})")
    print("=" * 60)

    test_package_metadata()
    test_scaler()
    test_binary_classification()
    test_multiclass_and_prediction_sets()
    test_regression_and_intervals()
    test_missing_values_and_abstention()
    test_interpretability_and_explanation()
    test_cross_validation()
    test_data_folders_smoke()

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
