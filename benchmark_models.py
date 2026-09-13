"""
benchmark_models.py
===================
Unified Multi-Dataset Benchmark Suite for AstralModel & Classic ML Baselines.

Compares AstralModel against comprehensive classic machine learning algorithms:
  - Linear/Logistic models (Ridge, Lasso, ElasticNet, Logistic Regression)
  - Tree ensembles (Random Forest, Gradient Boosting, HistGradientBoosting, AdaBoost)
  - Non-parametric / Kernel methods (SVM/SVC/SVR, K-Nearest Neighbours)
  - Probabilistic & Neural baselines (Gaussian Naive Bayes, MLP Neural Net)

Evaluates on all 3 repository datasets on both:
  1. RAW dataset (unscaled features)
  2. SCALED dataset (AstralScaler / StandardScaler normalized)

Datasets:
  - data-1: Ames House Prices (Regression — target: SalePrice)
  - data-2: EV Adoption (Binary Classification — target: Will_Buy_EV)
  - data-3: Cleveland Heart Disease (Binary Classification — target: AHD)

Outputs formatted comparison tables to console and writes results to:
  - data-1/benchmark_results.json
  - data-2/benchmark_results.json
  - data-3/benchmark_results.json
  - benchmark_results.json
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd

# Ensure unbuffered printing
import functools
print = functools.partial(print, flush=True)

# Suppress ConvergenceWarnings during fast benchmarking
warnings.filterwarnings("ignore")

# Ensure astral_model is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astral_model import AstralModel, AstralScaler, astral_train_test_split, __version__

# Classic ML imports from scikit-learn
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import LinearSVR, SVR, LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
)


SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS & PREPROCESSORS
# ═══════════════════════════════════════════════════════════════════════════════


def load_dataset_1():
    """Load and preprocess data-1: House Prices (Regression)."""
    path = os.path.join("data-1", "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(path)

    for id_col in ["Id", "id", "ID", "Unnamed: 0"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    target_col = "SalePrice"
    y = df[target_col].values.astype(np.float64)
    X_df = df.drop(columns=[target_col]).copy()

    feature_names = list(X_df.columns)
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.factorize(X_df[col].astype(str))[0].astype(float)
        if X_df[col].isna().any():
            med = float(X_df[col].median())
            X_df[col] = X_df[col].fillna(med if not np.isnan(med) else 0.0)

    X = X_df.values.astype(np.float64)
    return X, y, feature_names, "regression"


def load_dataset_2(max_rows=15000):
    """
    Load and preprocess data-2: EV Adoption (Classification).
    Uses a stratified subset (default 15,000) for fast cross-model evaluation.
    """
    path = os.path.join("data-2", "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")

    if max_rows and max_rows < 600000:
        df = pd.read_csv(path, nrows=max_rows * 2)
    else:
        df = pd.read_csv(path)

    for id_col in ["Id", "id", "ID", "Unnamed: 0"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    target_col = "Will_Buy_EV"
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    y = y_raw.isin(["yes", "1", "true"]).values.astype(int)

    X_df = df.drop(columns=[target_col]).copy()
    feature_names = list(X_df.columns)

    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.factorize(X_df[col].astype(str))[0].astype(float)
        if X_df[col].isna().any():
            med = float(X_df[col].median())
            X_df[col] = X_df[col].fillna(med if not np.isnan(med) else 0.0)

    X = X_df.values.astype(np.float64)

    # Stratified subsample to exactly max_rows if specified
    if max_rows and len(y) > max_rows:
        rng = np.random.RandomState(SEED)
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        ratio = max_rows / len(y)
        k0 = int(round(len(idx_0) * ratio))
        k1 = max_rows - k0
        chosen = np.concatenate([
            rng.choice(idx_0, size=min(k0, len(idx_0)), replace=False),
            rng.choice(idx_1, size=min(k1, len(idx_1)), replace=False),
        ])
        rng.shuffle(chosen)
        X = X[chosen]
        y = y[chosen]

    return X, y, feature_names, "classification"


def load_dataset_3():
    """Load and preprocess data-3: Cleveland Heart Disease (Classification)."""
    path = os.path.join("data-3", "Heart.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(path)

    for id_col in ["Id", "id", "ID", "Unnamed: 0"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    target_col = "AHD"
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    y = y_raw.isin(["yes", "1", "true"]).values.astype(int)

    X_df = df.drop(columns=[target_col]).copy()
    feature_names = list(X_df.columns)

    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.factorize(X_df[col].astype(str))[0].astype(float)
        if X_df[col].isna().any():
            med = float(X_df[col].median())
            X_df[col] = X_df[col].fillna(med if not np.isnan(med) else 0.0)

    X = X_df.values.astype(np.float64)
    return X, y, feature_names, "classification"



# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════


def get_regression_models(feature_names=None):
    """Return dictionary of regression models for benchmarking."""
    models = {
        "AstralModel": AstralModel(
            task="regression",
            feature_names=feature_names,
            ridge="auto",
            random_state=SEED,
            verbose=False,
        ),
        "Ridge Regression": Ridge(alpha=1.0, random_state=SEED),
        "Lasso Regression": Lasso(alpha=0.1, random_state=SEED, max_iter=1000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=1000),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=10, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=60, max_depth=4, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=60, random_state=SEED),
        "LinearSVR": LinearSVR(random_state=SEED, max_iter=500, tol=1e-3, dual="auto"),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=SEED),
        "MLP Regressor": MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=SEED),
    }
    return models


def get_classification_models(feature_names=None):
    """Return dictionary of classification models for benchmarking."""
    models = {
        "AstralModel": AstralModel(
            task="classification",
            feature_names=feature_names,
            ridge="auto",
            random_state=SEED,
            verbose=False,
        ),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=10, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=60, max_depth=4, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=60, random_state=SEED),
        "LinearSVC": LinearSVC(C=1.0, max_iter=400, tol=1e-3, dual="auto", random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=SEED),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=SEED),
    }
    return models


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model_regression(model, X_train, y_train, X_test, y_test):
    """Train and score a regression model."""
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 4)

    preds = model.predict(X_test)
    preds = np.nan_to_num(preds, nan=np.mean(y_train))

    r2 = float(r2_score(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))

    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "fit_time_s": fit_time,
    }


def evaluate_model_classification(model, X_train, y_train, X_test, y_test):
    """Train and score a classification model."""
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 4)

    preds = model.predict(X_test)
    preds = np.nan_to_num(preds, nan=0).astype(int)

    acc = float(accuracy_score(y_test, preds))
    f1_mac = float(f1_score(y_test, preds, average="macro", zero_division=0))
    f1_w = float(f1_score(y_test, preds, average="weighted", zero_division=0))

    # ROC-AUC if predict_proba or decision_function available
    auc = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] == 2:
                auc = float(roc_auc_score(y_test, probs[:, 1]))
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            auc = float(roc_auc_score(y_test, scores))
    except Exception:
        auc = None

    res = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_w, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "fit_time_s": fit_time,
    }
    return res


def run_benchmark_for_dataset(name, X, y, feature_names, task, test_size=0.20):
    """Run benchmark across raw and scaled versions of the dataset."""
    print(f"\n{'=' * 75}")
    print(f" BENCHMARK: {name} (Task: {task.upper()} | Samples: {len(y)} | Features: {X.shape[1]})")
    print(f"{'=' * 75}")

    # Split into train/test
    stratify = y if task == "classification" else None
    X_train_raw, X_test_raw, y_train, y_test = astral_train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=stratify
    )

    # Scale with AstralScaler
    scaler = AstralScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    raw_results = {}
    scaled_results = {}

    if task == "regression":
        models_raw = get_regression_models(feature_names)
        models_scaled = get_regression_models(feature_names)
        eval_fn = evaluate_model_regression
    else:
        models_raw = get_classification_models(feature_names)
        models_scaled = get_classification_models(feature_names)
        eval_fn = evaluate_model_classification

    # Run on Raw
    print("▶ Evaluating models on RAW (unscaled) dataset …")
    for m_name, model in models_raw.items():
        try:
            res = eval_fn(model, X_train_raw, y_train, X_test_raw, y_test)
            raw_results[m_name] = res
            print(f"  • {m_name:<24s} -> {res}")
        except Exception as e:
            raw_results[m_name] = {"error": str(e)}
            print(f"  • {m_name:<24s} -> Failed: {e}")

    # Run on Scaled
    print("\n▶ Evaluating models on SCALED dataset (AstralScaler) …")
    for m_name, model in models_scaled.items():
        try:
            res = eval_fn(model, X_train_scaled, y_train, X_test_scaled, y_test)
            scaled_results[m_name] = res
            print(f"  • {m_name:<24s} -> {res}")
        except Exception as e:
            scaled_results[m_name] = {"error": str(e)}
            print(f"  • {m_name:<24s} -> Failed: {e}")

    # Display comparison table
    print_comparison_table(name, task, raw_results, scaled_results)

    return {
        "dataset": name,
        "task": task,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "raw_results": raw_results,
        "scaled_results": scaled_results,
    }


def print_comparison_table(dataset_name, task, raw_results, scaled_results):
    """Print markdown-style comparison table."""
    print(f"\n─── COMPARISON SUMMARY: {dataset_name} ({task.upper()}) ───")
    if task == "regression":
        print(f"{'Model':<22s} | {'Raw R²':<8s} | {'Scaled R²':<9s} | {'Raw RMSE':<10s} | {'Scaled RMSE':<11s} | {'Scaled Time':<11s}")
        print("-" * 85)
        for name in raw_results:
            r = raw_results.get(name, {})
            s = scaled_results.get(name, {})
            r_r2 = f"{r.get('r2', 'N/A')}" if "r2" in r else "ERR"
            s_r2 = f"{s.get('r2', 'N/A')}" if "r2" in s else "ERR"
            r_rmse = f"{r.get('rmse', 'N/A')}" if "rmse" in r else "ERR"
            s_rmse = f"{s.get('rmse', 'N/A')}" if "rmse" in s else "ERR"
            s_time = f"{s.get('fit_time_s', 'N/A')}s" if "fit_time_s" in s else "ERR"
            print(f"{name:<22s} | {r_r2:<8s} | {s_r2:<9s} | {r_rmse:<10s} | {s_rmse:<11s} | {s_time:<11s}")
    else:
        print(f"{'Model':<22s} | {'Raw F1':<8s} | {'Scaled F1':<9s} | {'Raw Acc':<8s} | {'Scaled Acc':<10s} | {'Scaled AUC':<10s} | {'Time':<7s}")
        print("-" * 92)
        for name in raw_results:
            r = raw_results.get(name, {})
            s = scaled_results.get(name, {})
            r_f1 = f"{r.get('f1_macro', 'N/A')}" if "f1_macro" in r else "ERR"
            s_f1 = f"{s.get('f1_macro', 'N/A')}" if "f1_macro" in s else "ERR"
            r_acc = f"{r.get('accuracy', 'N/A')}" if "accuracy" in r else "ERR"
            s_acc = f"{s.get('accuracy', 'N/A')}" if "accuracy" in s else "ERR"
            s_auc = f"{s.get('roc_auc', 'N/A')}" if "roc_auc" in s and s['roc_auc'] is not None else "-"
            s_time = f"{s.get('fit_time_s', 'N/A')}s" if "fit_time_s" in s else "ERR"
            print(f"{name:<22s} | {r_f1:<8s} | {s_f1:<9s} | {r_acc:<8s} | {s_acc:<10s} | {s_auc:<10s} | {s_time:<7s}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="AstralModel Multi-Dataset Benchmark")
    parser.add_argument("--dataset", choices=["1", "2", "3", "all"], default="all",
                        help="Which dataset to benchmark (1=data-1, 2=data-2, 3=data-3, all=all 3)")
    parser.add_argument("--d2-samples", type=int, default=15000,
                        help="Sample size for data-2 EV Adoption dataset (default 15000)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"AstralModel v{__version__} — Comprehensive Multi-Dataset Benchmark")
    print("Evaluating AstralModel against 10 classic ML baselines on Raw vs Scaled datasets")
    print("=" * 80)

    all_benchmarks = {}

    # Dataset 3: Heart Disease (Classification)
    if args.dataset in ("3", "all"):
        try:
            X3, y3, f3, task3 = load_dataset_3()
            res3 = run_benchmark_for_dataset("data-3 (Heart Disease)", X3, y3, f3, task3)
            all_benchmarks["data-3"] = res3
            # Write inside data-3 folder
            with open(os.path.join("data-3", "benchmark_results.json"), "w") as f:
                json.dump(res3, f, indent=2)
            print("  -> Saved benchmark results to data-3/benchmark_results.json")
        except Exception as e:
            print(f"Error evaluating data-3: {e}")

    # Dataset 1: House Prices (Regression)
    if args.dataset in ("1", "all"):
        try:
            X1, y1, f1, task1 = load_dataset_1()
            res1 = run_benchmark_for_dataset("data-1 (House Prices)", X1, y1, f1, task1)
            all_benchmarks["data-1"] = res1
            # Write inside data-1 folder
            with open(os.path.join("data-1", "benchmark_results.json"), "w") as f:
                json.dump(res1, f, indent=2)
            print("  -> Saved benchmark results to data-1/benchmark_results.json")
        except Exception as e:
            print(f"Error evaluating data-1: {e}")

    # Dataset 2: EV Adoption (Classification)
    if args.dataset in ("2", "all"):
        try:
            X2, y2, f2, task2 = load_dataset_2(max_rows=args.d2_samples)
            res2 = run_benchmark_for_dataset(f"data-2 (EV Adoption, {len(y2)} samples)", X2, y2, f2, task2)
            all_benchmarks["data-2"] = res2
            # Write inside data-2 folder
            with open(os.path.join("data-2", "benchmark_results.json"), "w") as f:
                json.dump(res2, f, indent=2)
            print("  -> Saved benchmark results to data-2/benchmark_results.json")
        except Exception as e:
            print(f"Error evaluating data-2: {e}")

    # Write root summary
    with open("benchmark_results.json", "w") as f:
        json.dump(all_benchmarks, f, indent=2)
    print("\nSaved aggregated benchmark summary to benchmark_results.json")
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
