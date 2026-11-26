import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
print("Model type:", type(model).__name__)


# =====================================
# 3. LOAD DATASET
# =====================================

print("\nLoading dataset...")

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


# =====================================
# 4. PREPROCESS DATA
# =====================================

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

print("Preprocessing completed!")


# =====================================
# 5. EXTRACT FEATURE IMPORTANCE
# =====================================

print("\nCalculating feature importance...")

if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

elif hasattr(model, "coef_"):

    importance = np.abs(model.coef_)

    if importance.ndim > 1:
        importance = importance.mean(axis=0)

else:
    raise ValueError(
        "This model does not provide built-in feature importance."
    )


# =====================================
# 6. GET FEATURE NAMES
# =====================================

try:
    feature_names = preprocessor.get_feature_names_out()

except Exception:
    feature_names = [
        f"Feature_{i}" for i in range(len(importance))
    ]

if len(feature_names) != len(importance):
    feature_names = [
        f"Feature_{i}" for i in range(len(importance))
    ]

results = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

results = results.sort_values(
    by="Importance",
    ascending=False
)

# Save results
output_path = os.path.join(
    BASE_DIR, "feature_importance.csv"
)

results.to_csv(output_path, index=False)


# =====================================
# 7. DISPLAY TOP 20 FEATURES
# =====================================

print("\nTOP 20 IMPORTANT FEATURES")
print("-" * 60)

print(results.head(20).to_string(index=False))


# =====================================
# 8. VISUALIZATION
# =====================================

top_features = results.head(20).sort_values(
    by="Importance"
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"].astype(str),
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 20 Features for Cloud Security Threat Detection")

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "feature_importance.png"),
    dpi=300
)

plt.show()

print("\nAnalysis completed!")
print("CSV saved:", output_path)
print("Graph saved: feature_importance.png")