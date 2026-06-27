"""Train model.pkl from train.csv using the CardioScan notebook pipeline."""

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import solution

TRAIN_PATH = Path(__file__).resolve().parent / "train.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
SOLUTION_PATH = Path(__file__).resolve().parent / "solution.py"
OUTLIER_COLUMNS = ["trestbps", "chol", "thalach", "oldpeak"]


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


def build_pipeline(x):
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
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=0.36250727231041735,
                    class_weight="balanced",
                    penalty="l2",
                    solver="saga",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


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
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(x_train)
    print("Fitting pipeline...")
    pipeline.fit(x_train, y_train)

    from sklearn.metrics import roc_auc_score

    val_proba = pipeline.predict_proba(x_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"Validation ROC-AUC: {val_auc:.4f}")

    bundle = {
        "pipeline": pipeline,
        "outlier_bounds": bounds,
        "feature_columns": x.columns.tolist(),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
