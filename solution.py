# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation is computed end-to-end for preprocess/load_model/predict.
#
# Good luck!
# ----------------------------------------------------------------

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

MODEL_PATH = Path(__file__).resolve().parent / "model.bin"
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

SOURCE_COLUMNS = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

OUTPUT_FEATURE_COLUMNS = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
    "thal",
    "age_group_middle",
    "age_group_senior",
    "age_group_elderly",
    "high_chol",
    "high_bp",
    "sig_oldpeak",
    "hr_ratio",
    "risk_score",
    "cp_2",
    "cp_3",
    "cp_4",
    "restecg_1",
    "restecg_2",
    "slope_2",
    "slope_3",
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

    out["risk_score"] = (
        out["exang"].fillna(0).astype(float)
        + (out["oldpeak"] > 1.0).fillna(False).astype(float)
        + (out["cp"] == 4).fillna(False).astype(float)
        + (out["ca"] >= 1).fillna(False).astype(float)
        + (out["thal"] == 2).fillna(False).astype(float)
    )

    out["cp_2"] = (out["cp"] == 2).astype(int)
    out["cp_3"] = (out["cp"] == 3).astype(int)
    out["cp_4"] = (out["cp"] == 4).astype(int)
    out["restecg_1"] = (out["restecg"] == 1).astype(int)
    out["restecg_2"] = (out["restecg"] == 2).astype(int)
    out["slope_2"] = (out["slope"] == 2).astype(int)
    out["slope_3"] = (out["slope"] == 3).astype(int)
    out = out.drop(columns=["cp", "restecg", "slope"])

    return out


def _add_one_hot_and_risk(work):
    work["risk_score"] = (
        work["exang"].fillna(0).astype(float)
        + (work["oldpeak"] > 1.0).fillna(False).astype(float)
        + (work["cp"] == 4).fillna(False).astype(float)
        + (work["ca"] >= 1).fillna(False).astype(float)
        + (work["thal"] == 2).fillna(False).astype(float)
    )

    work["cp_2"] = (work["cp"] == 2).astype(np.int8)
    work["cp_3"] = (work["cp"] == 3).astype(np.int8)
    work["cp_4"] = (work["cp"] == 4).astype(np.int8)
    work["restecg_1"] = (work["restecg"] == 1).astype(np.int8)
    work["restecg_2"] = (work["restecg"] == 2).astype(np.int8)
    work["slope_2"] = (work["slope"] == 2).astype(np.int8)
    work["slope_3"] = (work["slope"] == 3).astype(np.int8)
    return work.drop(columns=["cp", "restecg", "slope"])


def _extract_raw(df):
    if SOURCE_COLUMNS[0] in df.columns:
        return df[SOURCE_COLUMNS].to_numpy(dtype=np.float64, na_value=np.nan)

    work = df.drop(columns=["Heart Disease", "id"], errors="ignore")
    rename_cols = {src: dst for src, dst in COLUMN_MAP.items() if src in work.columns}
    if rename_cols:
        work = work.rename(columns=rename_cols)
    work = work.replace("?", np.nan)
    return work[FEATURE_COLUMNS].to_numpy(dtype=np.float64, na_value=np.nan)


def _engineer_features(raw):
    age = raw[:, 0]
    cp = raw[:, 2]
    trestbps = np.clip(raw[:, 3], *OUTLIER_BOUNDS["trestbps"])
    chol = np.clip(raw[:, 4], *OUTLIER_BOUNDS["chol"])
    thalach = np.clip(raw[:, 7], *OUTLIER_BOUNDS["thalach"])
    oldpeak = np.clip(raw[:, 9], *OUTLIER_BOUNDS["oldpeak"])
    thal = raw[:, 12].copy()
    thal[thal == 3] = 0
    thal[thal == 6] = 1
    thal[thal == 7] = 2
    restecg = raw[:, 6]
    slope = raw[:, 10]
    ca = raw[:, 11]
    exang = raw[:, 8]
    ex = np.nan_to_num(exang, nan=0.0)

    n = raw.shape[0]
    x = np.empty((n, 25), dtype=np.float64)
    x[:, 0] = age
    x[:, 1] = raw[:, 1]
    x[:, 2] = trestbps
    x[:, 3] = chol
    x[:, 4] = raw[:, 5]
    x[:, 5] = thalach
    x[:, 6] = exang
    x[:, 7] = oldpeak
    x[:, 8] = ca
    x[:, 9] = thal
    x[:, 10] = (age >= 40) & (age < 55)
    x[:, 11] = (age >= 55) & (age < 70)
    x[:, 12] = age >= 70
    x[:, 13] = chol > 240
    x[:, 14] = trestbps > 140
    x[:, 15] = oldpeak > 2
    x[:, 16] = thalach / np.maximum(220 - age, 1)
    x[:, 17] = ex + (oldpeak > 1) + (cp == 4) + (ca >= 1) + (thal == 2)
    x[:, 18] = cp == 2
    x[:, 19] = cp == 3
    x[:, 20] = cp == 4
    x[:, 21] = restecg == 1
    x[:, 22] = restecg == 2
    x[:, 23] = slope == 2
    x[:, 24] = slope == 3
    return x


def preprocess(df):
    ids = df["id"].to_numpy()
    raw = _extract_raw(df).astype(np.float32, copy=False)
    out = pd.DataFrame({"id": ids})
    out.attrs["raw"] = raw
    return out


def _ensure_fused_weights(model):
    if "w" in model:
        return
    w = model["coef"] / model["scale_scale"]
    model["w"] = w.astype(np.float32)
    model["bias"] = np.float32(model["intercept"] - model["scale_mean"] @ w)
    model["impute_f32"] = model["impute_values"].astype(np.float32)


def load_model():
    global _CACHED_MODEL, OUTLIER_BOUNDS
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    with open(MODEL_PATH, "rb") as f:
        bundle = np.load(f, allow_pickle=True).item()
    if isinstance(bundle, dict):
        if "outlier_bounds" in bundle:
            OUTLIER_BOUNDS = bundle["outlier_bounds"]
        if "pipeline" in bundle:
            _CACHED_MODEL = bundle["pipeline"]
        else:
            _ensure_fused_weights(bundle)
            _CACHED_MODEL = bundle
    else:
        _CACHED_MODEL = bundle
    return _CACHED_MODEL


def _predict_fused(raw, model):
    impute = model["impute_f32"]
    w = model["w"]
    bias = model["bias"]

    age = raw[:, 0]
    cp = raw[:, 2]
    trestbps = np.clip(raw[:, 3], *OUTLIER_BOUNDS["trestbps"])
    chol = np.clip(raw[:, 4], *OUTLIER_BOUNDS["chol"])
    thalach = np.clip(raw[:, 7], *OUTLIER_BOUNDS["thalach"])
    oldpeak = np.clip(raw[:, 9], *OUTLIER_BOUNDS["oldpeak"])
    thal = raw[:, 12]
    thal = np.where(thal == 3, 0, np.where(thal == 6, 1, np.where(thal == 7, 2, thal)))
    restecg = raw[:, 6]
    slope = raw[:, 10]
    ca = raw[:, 11]
    exang = raw[:, 8]
    ex = np.nan_to_num(exang, nan=0.0)

    hr = np.where(
        np.isnan(thalach / np.maximum(220 - age, 1)),
        impute[16],
        thalach / np.maximum(220 - age, 1),
    )

    z = np.full(raw.shape[0], bias, dtype=np.float32)
    z += w[0] * np.where(np.isnan(age), impute[0], age)
    z += w[1] * np.where(np.isnan(raw[:, 1]), impute[1], raw[:, 1])
    z += w[2] * np.where(np.isnan(trestbps), impute[2], trestbps)
    z += w[3] * np.where(np.isnan(chol), impute[3], chol)
    z += w[4] * np.where(np.isnan(raw[:, 5]), impute[4], raw[:, 5])
    z += w[5] * np.where(np.isnan(thalach), impute[5], thalach)
    z += w[6] * np.where(np.isnan(exang), impute[6], exang)
    z += w[7] * np.where(np.isnan(oldpeak), impute[7], oldpeak)
    z += w[8] * np.where(np.isnan(ca), impute[8], ca)
    z += w[9] * np.where(np.isnan(thal), impute[9], thal)
    z += w[10] * ((age >= 40) & (age < 55))
    z += w[11] * ((age >= 55) & (age < 70))
    z += w[12] * (age >= 70)
    z += w[13] * (chol > 240)
    z += w[14] * (trestbps > 140)
    z += w[15] * (oldpeak > 2)
    z += w[16] * hr
    z += w[17] * (ex + (oldpeak > 1) + (cp == 4) + (ca >= 1) + (thal == 2))
    z += w[18] * (cp == 2)
    z += w[19] * (cp == 3)
    z += w[20] * (cp == 4)
    z += w[21] * (restecg == 1)
    z += w[22] * (restecg == 2)
    z += w[23] * (slope == 2)
    z += w[24] * (slope == 3)
    return 1.0 / (1.0 + np.exp(-z))


def _predict_numpy(df, model):
    cols = model["feature_columns"]
    _ensure_fused_weights(model)
    x = df[cols].to_numpy(dtype=np.float64, copy=True)
    x = np.where(np.isnan(x), model["impute_values"], x)
    logits = x @ model["w"].astype(np.float64) + float(model["bias"])
    return 1.0 / (1.0 + np.exp(-logits))


def predict(df, model):
    if isinstance(model, dict) and "coef" in model:
        raw = df.attrs.get("raw")
        if raw is not None:
            _ensure_fused_weights(model)
            preds = _predict_fused(raw, model)
        else:
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
