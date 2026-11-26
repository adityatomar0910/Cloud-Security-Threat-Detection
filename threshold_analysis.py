import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Register custom preprocessing function
def cast_to_string(X):
    return pd.DataFrame(X).astype(str)

sys.modules["__main__"].cast_to_string = cast_to_string


# =====================================
# 1. PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "TestOn", "merged_cloudtrail.json"
)


# =====================================
# 2. LOAD MODEL
# =====================================

print("Loading model...")

model = joblib.load(
    os.path.join(MODEL_DIR, "best_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

with open(
    os.path.join(MODEL_DIR, "preprocessing_info.json")
) as f:
    info = json.load(f)

print("Model loaded!")


# =====================================
# 3. LOAD DATASET
# =====================================

with open(DATASET_PATH) as f:
    data = json.load(f)

records = data["Records"] if isinstance(data, dict) else data

df = pd.json_normalize(records)

y_true = df["should_sample"].astype(int)

X = df.drop(
    columns=["should_sample", "attack_name"],
    errors="ignore"
)

# Remove unwanted columns
X = X.drop(
    columns=info.get("columns_to_remove", [])
    + info.get("identifier_columns", []),
    errors="ignore"
)

# Convert nested values
for col in X.columns:
    X[col] = X[col].apply(
        lambda v: json.dumps(v, sort_keys=True)
        if isinstance(v, (dict, list))
        else v
    )

# Extract time features
if "eventTime" in X.columns:
    dt = pd.to_datetime(
        X["eventTime"], errors="coerce", utc=True
    )

    X["event_hour"] = dt.dt.hour
    X["event_day"] = dt.dt.day
    X["event_month"] = dt.dt.month
    X["event_weekday"] = dt.dt.dayofweek

    X["is_weekend"] = (
        dt.dt.dayofweek >= 5
    ).astype(float)

    X = X.drop(columns=["eventTime"])

# Match training columns
X = X.reindex(columns=info["feature_columns"])

# Transform data
X_processed = preprocessor.transform(X)

print("Dataset preprocessing completed!")


# =====================================
# 4. GET SUSPICIOUS PROBABILITIES
# =====================================

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(X_processed)

    classes = list(model.classes_)

    suspicious_index = classes.index(1)

    scores = probabilities[:, suspicious_index]

elif hasattr(model, "decision_function"):

    raw_scores = model.decision_function(X_processed)

    scores = 1 / (
        1 + np.exp(-np.clip(raw_scores, -500, 500))
    )

else:
    raise ValueError(
        "Model does not support probability or decision scores."
    )


# =====================================
# 5. TEST DIFFERENT THRESHOLDS
# =====================================

print("\nTHRESHOLD COMPARISON")
print("-" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)

best_f1 = -1
best_threshold = None

for threshold in np.arange(0.1, 0.91, 0.1):

    predictions = (scores >= threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1:<15.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\nBest threshold by F1:", round(best_threshold, 2))
print("Best F1 score:", round(best_f1, 4))

print("\nThreshold analysis completed!")