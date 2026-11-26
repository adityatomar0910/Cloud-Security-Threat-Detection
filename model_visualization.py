import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# Register custom preprocessing function
def cast_to_string(X):
    return pd.DataFrame(X).astype(str)

sys.modules["__main__"].cast_to_string = cast_to_string

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "TestOn", "merged_cloudtrail.json"
)

# Load model
print("Loading model...")

model = joblib.load(
    os.path.join(MODEL_DIR, "best_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

with open(os.path.join(MODEL_DIR, "preprocessing_info.json")) as f:
    info = json.load(f)

# Load dataset
print("Loading dataset...")

with open(DATASET_PATH) as f:
    data = json.load(f)

records = data["Records"] if isinstance(data, dict) else data
df = pd.json_normalize(records)

y_true = df["should_sample"].astype(int)

X = df.drop(
    columns=["should_sample", "attack_name"],
    errors="ignore"
)

X = X.drop(
    columns=info.get("columns_to_remove", [])
    + info.get("identifier_columns", []),
    errors="ignore"
)

# Preprocessing
print("Preprocessing data...")

for col in X.columns:
    X[col] = X[col].apply(
        lambda v: json.dumps(v, sort_keys=True)
        if isinstance(v, (dict, list))
        else v
    )

if "eventTime" in X.columns:
    dt = pd.to_datetime(
        X["eventTime"], errors="coerce", utc=True
    )

    X["event_hour"] = dt.dt.hour
    X["event_day"] = dt.dt.day
    X["event_month"] = dt.dt.month
    X["event_weekday"] = dt.dt.day_of_week

    X["is_weekend"] = (
        dt.dt.day_of_week >= 5
    ).astype(float)

    X = X.drop(columns=["eventTime"])

X = X.reindex(columns=info["feature_columns"])
X_processed = preprocessor.transform(X)

# Predictions and scores
print("Generating predictions...")

y_pred = model.predict(X_processed)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_processed)
    classes = list(model.classes_)
    scores = probabilities[:, classes.index(1)]
elif hasattr(model, "decision_function"):
    scores = model.decision_function(X_processed)
else:
    raise ValueError("Model does not provide prediction scores.")

# Classification report
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true, y_pred,
    target_names=["Normal", "Suspicious"]
))

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Suspicious"],
    yticklabels=["Normal", "Suspicious"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# 3. Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(
    y_true, scores
)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300)
plt.show()

print("\nVisualization completed!")
print("Generated files:")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- precision_recall_curve.png")