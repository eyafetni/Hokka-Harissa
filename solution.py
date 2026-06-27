# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation is computed end-to-end for preprocess/load_model/predict.
#
# Good luck!
# ----------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

COLUMN_MAP = {
    "Age": "age",
    "Sex": "sex",
    "Chest pain type": "cp",
    "BP": "trestbps",
    "Cholesterol": "chol",
    "FBS over 120": "fbs",
    "EKG results": "restecg",
    "Max HR": "thalach",
    "Exercise angina": "exang",
    "ST depression": "oldpeak",
    "Slope of ST": "slope",
    "Number of vessels fluro": "ca",
    "Thallium": "thal",
}

# Competition Thallium (3/6/7) -> Kaggle-style encoding (0/1/2)
THALLIUM_MAP = {3: 0, 6: 1, 7: 2, 3.0: 0, 6.0: 1, 7.0: 2}

# Fitted on train.csv during model training (see train_model.py)
OUTLIER_BOUNDS = {
    "trestbps": (60.0, 200.0),
    "chol": (81.0, 410.0),
    "thalach": (70.0, 238.0),
    "oldpeak": (-4.2, 5.6),
}

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
_CACHED_MODEL = None


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def _sanitize_values(df):
    return df.replace("?", np.nan)


def _to_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _map_competition_columns(df):
    out = df.copy()
    rename_cols = {src: dst for src, dst in COLUMN_MAP.items() if src in out.columns}
    out = out.rename(columns=rename_cols)

    if "thal" in out.columns:
        out["thal"] = out["thal"].map(THALLIUM_MAP)

    return out


def _cap_outliers(df, bounds=None):
    bounds = bounds or OUTLIER_BOUNDS
    out = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def engineer_features(df):
    out = df.copy()

    out["age_group_middle"] = ((out["age"] >= 40) & (out["age"] < 55)).astype(int)
    out["age_group_senior"] = ((out["age"] >= 55) & (out["age"] < 70)).astype(int)
    out["age_group_elderly"] = (out["age"] >= 70).astype(int)

    out["high_chol"] = (out["chol"] > 240).astype(int)
    out["high_bp"] = (out["trestbps"] > 140).astype(int)
    out["sig_oldpeak"] = (out["oldpeak"] > 2.0).astype(int)
    out["hr_ratio"] = out["thalach"] / (220 - out["age"]).clip(lower=1)

    return out


def preprocess(df):
    ids = df["id"]
    work = df.drop(columns=["Heart Disease", "id"], errors="ignore")

    rename_cols = {src: dst for src, dst in COLUMN_MAP.items() if src in work.columns}
    if rename_cols:
        work = work.rename(columns=rename_cols)

    work = work.replace("?", np.nan)

    for col in FEATURE_COLUMNS:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "thal" in work.columns:
        work["thal"] = work["thal"].map(THALLIUM_MAP)

    for col, (lower, upper) in OUTLIER_BOUNDS.items():
        if col in work.columns:
            work[col] = work[col].clip(lower=lower, upper=upper)

    work["age_group_middle"] = ((work["age"] >= 40) & (work["age"] < 55)).astype(np.int8)
    work["age_group_senior"] = ((work["age"] >= 55) & (work["age"] < 70)).astype(np.int8)
    work["age_group_elderly"] = (work["age"] >= 70).astype(np.int8)
    work["high_chol"] = (work["chol"] > 240).astype(np.int8)
    work["high_bp"] = (work["trestbps"] > 140).astype(np.int8)
    work["sig_oldpeak"] = (work["oldpeak"] > 2.0).astype(np.int8)
    work["hr_ratio"] = work["thalach"] / (220 - work["age"]).clip(lower=1)

    work["id"] = ids.to_numpy()
    return work


def load_model():
    global _CACHED_MODEL, OUTLIER_BOUNDS
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        if "outlier_bounds" in bundle:
            OUTLIER_BOUNDS = bundle["outlier_bounds"]
        if "pipeline" in bundle:
            _CACHED_MODEL = bundle["pipeline"]
        else:
            _CACHED_MODEL = bundle
    else:
        _CACHED_MODEL = bundle
    return _CACHED_MODEL


def _predict_numpy(df, model):
    cols = model["feature_columns"]
    impute = model["impute_values"]
    x = df[cols].to_numpy(dtype=np.float64, copy=True)
    for j in range(x.shape[1]):
        mask = np.isnan(x[:, j])
        if mask.any():
            x[mask, j] = impute[j]
    x -= model["scale_mean"]
    x /= model["scale_scale"]
    logits = x @ model["coef"] + model["intercept"]
    return 1.0 / (1.0 + np.exp(-logits))


def predict(df, model):
    if isinstance(model, dict) and "coef" in model:
        preds = _predict_numpy(df, model)
    else:
        feature_cols = [col for col in df.columns if col != "id"]
        preds = model.predict_proba(df[feature_cols])[:, 1]

    return pd.DataFrame({"id": df["id"], "Heart Disease": preds})


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------


def run(df) -> tuple[float, float, float]:
    from time import time

    start = time.perf_counter()

    df_processed = preprocess(df)
    model = load_model()
    size = get_model_size(model)
    predictions = predict(df_processed, model)

    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
