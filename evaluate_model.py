import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Register custom function used when saving the preprocessor
def cast_to_string(X):
    return pd.DataFrame(X).astype(str)

sys.modules["__main__"].cast_to_string = cast_to_string


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "TestOn",
    "merged_cloudtrail.json"
)


# ==========================================
# 2. LOAD MODEL FILES
# ==========================================

print("\nLoading trained model...")

model = joblib.load(
    os.path.join(MODEL_DIR, "best_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

with open(
    os.path.join(MODEL_DIR, "preprocessing_info.json"),
    "r"
) as file:
    preprocessing_info = json.load(file)

print("Model loaded successfully!")


# ==========================================
# 3. LOAD TEST DATASET
# ==========================================

print("\nLoading test dataset...")

with open(DATASET_PATH, "r") as file:
    data = json.load(file)

if isinstance(data, dict):
    records = data.get("Records", data.get("records"))
else:
    records = data

if not records:
    raise ValueError("No records found in dataset.")

df = pd.json_normalize(records)

if "should_sample" not in df.columns:
    raise ValueError(
        "Actual labels (should_sample) are missing from the dataset."
    )

# Actual labels: False = Normal, True = Suspicious
y_true = df["should_sample"].astype(int)

print("Total test records:", len(df))


# ==========================================
# 4. PREPARE TEST DATA
# ==========================================

print("\nPreprocessing test data...")

X = df.drop(
    columns=["should_sample", "attack_name"],
    errors="ignore"
)

columns_to_remove = preprocessing_info.get(
    "columns_to_remove", []
)

identifier_columns = preprocessing_info.get(
    "identifier_columns",
    ["eventID", "requestID", "sharedEventID"]
)

feature_columns = preprocessing_info["feature_columns"]

X.drop(
    columns=columns_to_remove + identifier_columns,
    errors="ignore",
    inplace=True
)


# Convert nested JSON values
for col in X.columns:
    X[col] = X[col].apply(
        lambda value: json.dumps(value, sort_keys=True)
        if isinstance(value, (dict, list))
        else value
    )


# Extract time features
if "eventTime" in X.columns:

    dt = pd.to_datetime(
        X["eventTime"],
        errors="coerce",
        utc=True
    )

    X["event_hour"] = dt.dt.hour
    X["event_day"] = dt.dt.day
    X["event_month"] = dt.dt.month
    X["event_weekday"] = dt.dt.dayofweek

    X["is_weekend"] = (
        dt.dt.dayofweek >= 5
    ).astype(float)

    X.drop(columns=["eventTime"], inplace=True)


# Match training feature columns
X = X.reindex(columns=feature_columns)

# Apply saved preprocessing pipeline
X_processed = preprocessor.transform(X)

print("Preprocessing completed!")


# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_pred = np.asarray(
    model.predict(X_processed)
).reshape(-1).astype(int)

if len(y_true) != len(y_pred):
    raise ValueError("Actual labels and predictions have different lengths.")


# ==========================================
# 6. EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 45)
print("       CLOUD SECURITY MODEL EVALUATION")
print("=" * 45)

print("Total Records:", len(y_true))

print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"False Positive Rate:{fpr:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Suspicious"],
        zero_division=0
    )
)

print("\nConfusion Matrix Details:")
print("True Normal:", tn)
print("False Suspicious:", fp)
print("Missed Suspicious:", fn)
print("Correct Suspicious:", tp)

print("\nEvaluation completed!")