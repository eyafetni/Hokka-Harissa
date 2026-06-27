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

    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 40, 55, 70, 120],
        labels=["young", "middle", "senior", "elderly"],
    )
    age_dummies = pd.get_dummies(out["age_group"], prefix="age_group").astype(int)
    out = pd.concat([out, age_dummies], axis=1).drop(columns=["age_group"])

    for col in ["age_group_middle", "age_group_senior", "age_group_elderly"]:
        if col not in out.columns:
            out[col] = 0

    out["high_chol"] = (out["chol"] > 240).astype(int)
    out["high_bp"] = (out["trestbps"] > 140).astype(int)
    out["sig_oldpeak"] = (out["oldpeak"] > 2.0).astype(int)
    out["hr_ratio"] = out["thalach"] / (220 - out["age"])

    return out


def preprocess(df):
    ids = df["id"].copy()
    work = df.drop(columns=["Heart Disease", "id"], errors="ignore")
    work = _map_competition_columns(work)
    work = _sanitize_values(work)
    work = _to_numeric(work, FEATURE_COLUMNS)
    work = _cap_outliers(work)
    work = engineer_features(work)
    work.insert(0, "id", ids)
    return work


def load_model():
    global OUTLIER_BOUNDS
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        if "outlier_bounds" in bundle:
            OUTLIER_BOUNDS = bundle["outlier_bounds"]
        return bundle["pipeline"]
    return bundle


def predict(df, model):
    feature_cols = [col for col in df.columns if col != "id"]
    preds = model.predict_proba(df[feature_cols])[:, 1]
    predictions = pd.DataFrame({"id": df["id"], "Heart Disease": preds})
    return predictions


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
