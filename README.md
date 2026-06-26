# AINS Data Science Challenge

## Files

| File          | Description                                                                                  |
| :------------ | :------------------------------------------------------------------------------------------- |
| `train.csv`   | Training set with features **and** the target column `Heart Disease`.                        |
| `solution.py` | Template with the three functions you must implement: `preprocess`, `load_model`, `predict`. |

---

## Objective

Build a model that predicts the probability of **heart disease** being `Presence` for each patient.

This is a **binary classification** problem. The target column is `Heart Disease`, with values:

| Value      | Meaning                                   |
| :--------- | :---------------------------------------- |
| `Presence` | Heart disease is present (positive class) |
| `Absence`  | Heart disease is absent (negative class)  |

Your `predict()` function must return a probability between `0` and `1` for the `Presence` class.

---

## Columns

### Identifier & Target

- `id` — Unique patient identifier.
- `Heart Disease` — **[TARGET]** `Presence` or `Absence`. Present in `train.csv` only.

### Features

| Column                    | Description                                                                                                    |
| :------------------------ | :------------------------------------------------------------------------------------------------------------- |
| `Age`                     | Age in years.                                                                                                  |
| `Sex`                     | Biological sex (`1` = male, `0` = female).                                                                     |
| `Chest pain type`         | Chest pain category (`1` = typical angina, `2` = atypical angina, `3` = non-anginal pain, `4` = asymptomatic). |
| `BP`                      | Resting blood pressure in mm Hg.                                                                               |
| `Cholesterol`             | Serum cholesterol in mg/dl.                                                                                    |
| `FBS over 120`            | Fasting blood sugar greater than 120 mg/dl (`1` = true, `0` = false).                                          |
| `EKG results`             | Resting electrocardiographic results (`0`, `1`, or `2`).                                                       |
| `Max HR`                  | Maximum heart rate achieved during exercise.                                                                   |
| `Exercise angina`         | Exercise-induced angina (`1` = yes, `0` = no).                                                                 |
| `ST depression`           | ST depression induced by exercise relative to rest.                                                            |
| `Slope of ST`             | Slope of the peak exercise ST segment (`1` = upsloping, `2` = flat, `3` = downsloping).                        |
| `Number of vessels fluro` | Number of major vessels colored by fluoroscopy (`0` to `3`).                                                   |
| `Thallium`                | Thallium stress test result (`3` = normal, `6` = fixed defect, `7` = reversible defect).                       |

---

## Scoring

Your **final score** combines three factors:

```
score = auc_roc × size_penalty × duration_penalty

where:
  auc_roc          = Area Under the Receiver Operating Characteristic Curve
  size_penalty     = max(0.5, 1 − model_size_mb / 200)
  duration_penalty = max(0.5, 1 − pipeline_seconds / 10)
```

### AUC-ROC

AUC-ROC measures how well your model ranks positive cases (`Presence`) above negative cases (`Absence`) across all probability thresholds. A perfect model scores `1.0`; random guessing scores `0.5`.

### Size Penalty

Penalizes large model files. A 0 MB model scores `1.0` (no penalty); a 100 MB model scores `0.5`. The penalty floors at `0.5`, so no model loses more than half its AUC-ROC.

### Latency Penalty

Penalizes slow end-to-end execution. 0 s → `1.0`; 5 s → `0.5`. Floors at `0.5`. The timer includes `preprocess()`, `load_model()`, and `predict()`.

> **Tip:** Don't just chase AUC-ROC. A lightweight, fast model with slightly lower AUC-ROC can outscore a bloated one.

---

## Submission Format

Your `.zip` must contain exactly **3 files** at the root (no nested folders):

| File               | Required | Notes                                                                  |
| :----------------- | :------: | :--------------------------------------------------------------------- |
| `solution.py`      |   yes    | Must export `preprocess`, `load_model`, `predict`.                     |
| `model.*`          |   yes    | Exactly one file starting with `model` (e.g. `model.pkl`, `model.pt`). |
| `requirements.txt` |   yes    | Extra pip packages. Can be empty but must exist.                       |

### What `predict()` must return

A **pandas DataFrame** with two columns:

```
id,Heart Disease
0,0.85
1,0.23
2,0.91
...
```

Every `id` from the test set must be present. The `Heart Disease` column must contain probabilities between `0` and `1` for the `Presence` class.

---

## Pre-installed Packages

The evaluation container already has these installed, you don't need to list them in `requirements.txt`:

| Package      | Version     |
| :----------- | :---------- |
| numpy        | 1.26.4      |
| scipy        | 1.11.4      |
| pandas       | 2.1.4       |
| scikit-learn | 1.3.2       |
| xgboost      | 2.0.3       |
| lightgbm     | 4.6.0       |
| catboost     | 1.2.3       |
| torch        | 2.6.0 (CPU) |
| torchvision  | 0.21.0      |
| torchaudio   | 2.6.0       |
| tensorflow   | 2.14.0      |
| joblib       | 1.3.2       |

You **can** add extra packages via `requirements.txt`. You **cannot** override pre-installed versions.

---

## Limits

| Constraint           | Value  |
| :------------------- | :----- |
| Max upload size      | 50 MB  |
| Container memory     | 4 GB   |
| Container CPU        | 1 core |
| Execution timeout    | 180 s  |
| Submissions per team | 20     |

Good luck!
