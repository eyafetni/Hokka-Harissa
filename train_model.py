"""Train model.pkl from train.csv using the CardioScan notebook pipeline."""

import re
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import solution

TRAIN_PATH = Path(__file__).resolve().parent / "train.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
SOLUTION_PATH = Path(__file__).resolve().parent / "solution.py"
OUTLIER_COLUMNS = ["trestbps", "chol", "thalach", "oldpeak"]
TUNING_SAMPLE_SIZE = 80_000
RANDOM_STATE = 42
SKIP_TUNING = True
CACHED_MODEL_PARAMS = {
    "C": 15.112022770860973,
    "class_weight": "balanced",
    "penalty": "l2",
    "solver": "saga",
}


def compute_outlier_bounds(df, columns, factor=3.0):
    bounds = {}
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (float(q1 - factor * iqr), float(q3 + factor * iqr))
    return bounds


def update_solution_bounds(bounds):
    text = SOLUTION_PATH.read_text(encoding="utf-8")
    block = "OUTLIER_BOUNDS = {\n" + "".join(
        f'    "{col}": ({lo:.1f}, {hi:.1f}),\n' for col, (lo, hi) in bounds.items()
    ) + "}"
    text, count = re.subn(
        r"OUTLIER_BOUNDS = \{[^}]+\}",
        block,
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update OUTLIER_BOUNDS in solution.py")
    SOLUTION_PATH.write_text(text, encoding="utf-8")
    solution.OUTLIER_BOUNDS.update(bounds)


def deduplicate_training_rows(df):
    mapped = solution._map_competition_columns(
        df.drop(columns=["Heart Disease"], errors="ignore")
    )
    feature_cols = [col for col in solution.FEATURE_COLUMNS if col in mapped.columns]
    before = len(df)
    keep_idx = mapped.drop_duplicates(subset=feature_cols).index
    deduped = df.loc[keep_idx].reset_index(drop=True)
    removed = before - len(deduped)
    if removed:
        print(f"Removed {removed:,} duplicate feature rows")
    return deduped


def prepare_training_frame(df):
    y = (df["Heart Disease"] == "Presence").astype(int)
    processed = solution.preprocess(df)
    x = processed.drop(columns=["id"])
    return x, y


def build_preprocessor(x):
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    cat_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", cat_transformer, cat_features),
        ]
    )


def build_pipeline(x, model_params=None):
    model_params = model_params or {}
    defaults = {
        "random_state": RANDOM_STATE,
        "max_iter": 5000,
    }
    defaults.update(model_params)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(x)),
            ("model", LogisticRegression(**defaults)),
        ]
    )


def export_inference_bundle(pipeline, feature_columns, bounds):
    preprocessor = pipeline.named_steps["preprocessor"]
    lr = pipeline.named_steps["model"]
    num_pipe = preprocessor.named_transformers_["num"]
    return {
        "feature_columns": feature_columns,
        "impute_values": num_pipe.named_steps["imputer"].statistics_,
        "scale_mean": num_pipe.named_steps["scaler"].mean_,
        "scale_scale": num_pipe.named_steps["scaler"].scale_,
        "coef": lr.coef_.ravel(),
        "intercept": float(lr.intercept_[0]),
        "outlier_bounds": bounds,
    }


def stratified_subsample(x, y, n_samples, random_state=RANDOM_STATE):
    if len(x) <= n_samples:
        return x, y
    x_sub, _, y_sub, _ = train_test_split(
        x,
        y,
        train_size=n_samples,
        random_state=random_state,
        stratify=y,
    )
    return x_sub, y_sub


def tune_hyperparameters(x_train, y_train):
    preprocessor = build_preprocessor(x_train)
    base_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)),
        ]
    )

    param_dist = {
        "model__C": uniform(0.001, 20),
        "model__solver": ["lbfgs", "liblinear", "saga"],
        "model__class_weight": [None, "balanced"],
        "model__penalty": ["l2", "l1"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    x_tune, y_tune = stratified_subsample(x_train, y_train, TUNING_SAMPLE_SIZE)

    print(
        f"Tuning on {len(x_tune):,} stratified samples "
        f"(from {len(x_train):,} train rows)..."
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        rand_search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=param_dist,
            n_iter=100,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        rand_search.fit(x_tune, y_tune)

    print(f"Best CV ROC-AUC: {rand_search.best_score_:.4f}")
    print("Best parameters:")
    for key, value in rand_search.best_params_.items():
        print(f"  {key.replace('model__', ''):20s}: {value}")

    return rand_search.best_params_


def extract_model_params(best_params):
    return {
        key.replace("model__", ""): value for key, value in best_params.items()
    }


def verify_prediction_parity(pipeline, bundle, x_val, ids):
    sklearn_preds = pipeline.predict_proba(x_val)[:, 1]
    val_df = x_val.copy()
    val_df["id"] = ids
    fast_preds = solution.predict(val_df, bundle)["Heart Disease"].to_numpy()
    max_diff = float(np.max(np.abs(sklearn_preds - fast_preds)))
    print(f"Prediction parity max abs diff: {max_diff:.2e}")
    if max_diff >= 1e-10:
        raise RuntimeError(f"Fast predict diverges from sklearn pipeline: {max_diff}")
    return max_diff


def benchmark_inference(df, eval_auc):
    solution._CACHED_MODEL = None
    start = perf_counter()
    processed = solution.preprocess(df)
    model = solution.load_model()
    solution.predict(processed, model)
    duration = perf_counter() - start
    return estimate_composite_score(eval_auc, MODEL_PATH, duration)


def estimate_composite_score(auc, model_path, duration_seconds):
    size_mb = model_path.stat().st_size / (1024 * 1024)
    size_penalty = max(0.5, 1 - size_mb / 200)
    duration_penalty = max(0.5, 1 - duration_seconds / 10)
    composite = auc * size_penalty * duration_penalty
    print(f"Model size: {size_mb:.4f} MB (penalty {size_penalty:.4f})")
    print(f"Pipeline duration: {duration_seconds:.4f} s (penalty {duration_penalty:.4f})")
    print(f"Estimated composite score: {composite:.4f}")
    return composite


def main():
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH)
    df = deduplicate_training_rows(df)

    raw = solution._map_competition_columns(
        df.drop(columns=["Heart Disease"], errors="ignore")
    )
    raw = solution._sanitize_values(raw)
    raw = solution._to_numeric(raw, solution.FEATURE_COLUMNS)
    bounds = compute_outlier_bounds(raw, OUTLIER_COLUMNS)
    update_solution_bounds(bounds)
    print("Outlier bounds:", bounds)

    x, y = prepare_training_frame(df)
    print(f"Training samples: {len(x):,} | features: {x.shape[1]}")

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    val_ids = np.arange(len(x_val))

    if SKIP_TUNING:
        print("Using cached tuned hyperparameters (skipping RandomizedSearchCV)...")
        model_params = CACHED_MODEL_PARAMS
    else:
        best_params = tune_hyperparameters(x_train, y_train)
        model_params = extract_model_params(best_params)

    print("Refitting best pipeline on full training data...")
    pipeline = build_pipeline(x, model_params)
    pipeline.fit(x, y)

    val_proba = pipeline.predict_proba(x_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"Holdout ROC-AUC: {val_auc:.4f}")

    bundle = export_inference_bundle(pipeline, x.columns.tolist(), bounds)
    verify_prediction_parity(pipeline, bundle, x_val, val_ids)

    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved lightweight bundle: {MODEL_PATH}")

    print(f"Benchmarking full dataset ({len(df):,} rows)...")
    benchmark_inference(df, val_auc)


if __name__ == "__main__":
    main()
