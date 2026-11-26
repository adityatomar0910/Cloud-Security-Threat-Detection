import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import sys
import tensorflow as tf


def cast_to_string(X):
    return pd.DataFrame(X).astype(str)


sys.modules["__main__"].cast_to_string = cast_to_string


# Streamlit page configuration
st.set_page_config(
    page_title="Cloud Security Threat Detection",
    page_icon="🔐",
    layout="wide"
)

# ==================================================
# MODEL FILE PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

ANN_MODEL_PATH = os.path.join(MODEL_DIR, "ann_model.keras")

PREPROCESSING_INFO_PATH = os.path.join(
    MODEL_DIR, "preprocessing_info.json"
)

MODEL_INFO_PATH = os.path.join(
    MODEL_DIR, "model_info.json"
)


@st.cache_resource
def load_project_files():

    required_files = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        PREPROCESSING_INFO_PATH
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(PREPROCESSING_INFO_PATH, "r") as file:
        preprocessing_info = json.load(file)

    model_info = {}

    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, "r") as file:
            model_info = json.load(file)

    # Load ANN model
    ann_model = None

    if os.path.exists(ANN_MODEL_PATH):
        try:
            import tensorflow as tf

            ann_model = tf.keras.models.load_model(
                ANN_MODEL_PATH,
                compile=False
            )

        except Exception as error:
            st.warning(
                f"ANN model unavailable: {error}"
            )

    return model, preprocessor, preprocessing_info, model_info, ann_model


# =========================================================
# 4. LOAD TRAINED MODEL AND PREPROCESSOR
# =========================================================



# =========================================================
# 5. DATA PREPROCESSING FUNCTIONS
# =========================================================

def convert_nested_values(df):

    df = df.copy()

    for col in df.columns:
        df[col] = df[col].apply(
            lambda value: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )

    return df


def extract_time_features(df):

    df = df.copy()

    if "eventTime" in df.columns:

        dt = pd.to_datetime(
            df["eventTime"],
            errors="coerce",
            utc=True
        )

        df["event_hour"] = dt.dt.hour
        df["event_day"] = dt.dt.day
        df["event_month"] = dt.dt.month
        df["event_weekday"] = dt.dt.dayofweek

        df["is_weekend"] = (
            dt.dt.dayofweek >= 5
        ).astype(float)

        df.drop(columns=["eventTime"], inplace=True)

    return df


# =========================================================
# 6. PREPARE CLOUDTRAIL DATA
# =========================================================

def prepare_data(records, preprocessor, preprocessing_info):

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("No CloudTrail records found.")

    df = pd.json_normalize(records)

    if df.empty:
        raise ValueError("Uploaded JSON contains no usable records.")

    # Remove target columns if present
    df.drop(
        columns=["should_sample", "attack_name"],
        errors="ignore",
        inplace=True
    )

    columns_to_remove = preprocessing_info.get(
        "columns_to_remove", []
    )

    identifier_columns = preprocessing_info.get(
        "identifier_columns",
        ["eventID", "requestID", "sharedEventID"]
    )

    feature_columns = preprocessing_info.get(
        "feature_columns", []
    )

    if not feature_columns:
        raise ValueError(
            "Feature columns are missing from preprocessing_info.json."
        )

    # Remove excluded columns
    df.drop(
        columns=columns_to_remove + identifier_columns,
        errors="ignore",
        inplace=True
    )

    # Convert nested JSON objects
    df = convert_nested_values(df)

    # Extract time features
    df = extract_time_features(df)

    # Match training feature columns and their order
    df = df.reindex(columns=feature_columns)

    # Apply fitted preprocessing pipeline
    model_input = original_data.copy()

# Extract time-based features
if "eventTime" in model_input.columns:

    event_datetime = pd.to_datetime(
        model_input = original_data.copy()
        errors="coerce",
        utc=True
    )

    model_input["event_hour"] = event_datetime.dt.hour
    model_input["event_day"] = event_datetime.dt.day
    model_input["event_month"] = event_datetime.dt.month
    model_input["event_weekday"] = event_datetime.dt.dayofweek
    model_input["is_weekend"] = (
        event_datetime.dt.dayofweek >= 5
    ).astype(float)

# Match training columns
expected_columns = list(preprocessor.feature_names_in_)

for column in expected_columns:
    if column not in model_input.columns:
        model_input[column] = np.nan

model_input = model_input[expected_columns]

processed_data = preprocessor.transform(model_input)



# =========================================================
# 7. GENERATE PREDICTIONS
# =========================================================

def make_predictions(model, processed_data, threshold=0.5):

    probabilities = None

    if hasattr(model, "predict_proba"):

        all_probabilities = model.predict_proba(processed_data)

        classes = list(model.classes_)

        if 1 in classes:
            suspicious_index = classes.index(1)
            probabilities = all_probabilities[:, suspicious_index]
        else:
            probabilities = all_probabilities[:, -1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(processed_data)

        probabilities = 1 / (
            1 + np.exp(-np.clip(scores, -500, 500))
        )

        predictions = (scores >= 0).astype(int)

    else:

        predictions = model.predict(processed_data)

        predictions = np.asarray(predictions).reshape(-1).astype(int)

    return predictions, probabilities

# =========================================================
# LOAD PROJECT FILES
# =========================================================

try:

    model, preprocessor, preprocessing_info, model_info, ann_model = (
        load_project_files()
    )

    st.success("Model and preprocessor loaded successfully!")

except Exception as error:

    st.error("Unable to load project files.")
    st.exception(error)
    st.stop()

# =========================================================
# 8. DASHBOARD HEADER
# =========================================================

st.title("🔐 Cloud Security Threat Detection")

st.markdown(
    """
    ### Detect suspicious activities in AWS CloudTrail logs using Machine Learning.

    Upload an AWS CloudTrail JSON file to analyze cloud activity
    and identify potentially suspicious events.
    """
)
st.divider()

# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model Name",
    model_info.get("model", "Random Forest")
)

col2.metric(
    "Detection Threshold",
    f"{model_info.get('threshold', 0.5):.2f}"
)

col3.metric(
    "Test Accuracy",
    f"{model_info.get('accuracy', 0) * 100:.2f}%"
)

st.divider()
# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "91.80%")
col2.metric("Precision", "96.92%")
col3.metric("Recall", "24.33%")
col4.metric("F1 Score", "38.90%")

st.divider()


# =========================================================
# 10. MODEL INFORMATION
# =========================================================

with st.expander("Model Information"):

    st.write(
        "Selected Model:",
        model_info.get(
            "best_model_name",
            model_info.get("model_name", "Saved ML Model")
        )
    )

    if model_info:
        st.json(model_info)
    else:
        st.info("No model information available.")


st.divider()


# =========================================================
# 11. UPLOAD CLOUDTRAIL JSON FILE
# =========================================================

st.subheader("Upload AWS CloudTrail Logs")
st.subheader("Select Detection Model")

available_models = ["Random Forest"]

if ann_model is not None:
    available_models.append("Artificial Neural Network (ANN)")

selected_model = st.selectbox(
    "Choose a model",
    available_models
)

uploaded_file = st.file_uploader(
    "Choose a CloudTrail JSON file",
    type=["json"]
)


# =========================================================
# 12. ANALYZE UPLOADED FILE
# =========================================================

if uploaded_file is not None:

    try:

        uploaded_data = json.load(uploaded_file)

        # Read CloudTrail records
        if isinstance(uploaded_data, dict):

            if "Records" not in uploaded_data:
                st.error("Invalid CloudTrail JSON format.")
                st.stop()

            records = uploaded_data["Records"]

        elif isinstance(uploaded_data, list):

            records = uploaded_data

        else:

            st.error("Unsupported JSON format.")
            st.stop()

        if not isinstance(records, list) or len(records) == 0:
            st.error("No CloudTrail records found.")
            st.stop()

        st.success(
            f"Successfully uploaded {len(records):,} records."
        )

        # Convert records into DataFrame
        original_data = pd.json_normalize(records)

   # PREPROCESS UPLOADED DATA

with st.spinner("Processing CloudTrail records..."):

    model_input = original_data.copy()

    if "eventTime" in model_input.columns:

        event_datetime = pd.to_datetime(
            model_input["eventTime"],
            errors="coerce",
            utc=True
        )

        model_input["event_hour"] = event_datetime.dt.hour
        model_input["event_day"] = event_datetime.dt.day
        model_input["event_month"] = event_datetime.dt.month
        model_input["event_weekday"] = event_datetime.dt.dayofweek

        model_input["is_weekend"] = (
            event_datetime.dt.dayofweek >= 5
        ).astype(float)

    expected_columns = list(preprocessor.feature_names_in_)

    for column in expected_columns:
        if column not in model_input.columns:
            model_input[column] = np.nan

    model_input = model_input[expected_columns]

    processed_data = preprocessor.transform(model_input)

        # -------------------------------------------------
        # GENERATE PREDICTIONS
        # -------------------------------------------------

        with st.spinner("Detecting suspicious activities..."):

            if selected_model == "Random Forest":

                predictions, probabilities = make_predictions(
                    model,
                    processed_data,
                    threshold=model_info.get("threshold", 0.60)
                )

            else:

                ann_input = (
                    processed_data.toarray()
                    if hasattr(processed_data, "toarray")
                    else np.asarray(processed_data)
                )

                ann_input = ann_input.astype("float32")
                ann_input = np.clip(ann_input, -10, 10)

                probabilities = ann_model.predict(
                    ann_input,
                    verbose=0
                ).ravel()

                predictions = (
                    probabilities >= 0.5
                ).astype(int)

        # -------------------------------------------------
        # CREATE RESULTS
        # -------------------------------------------------

        results = pd.DataFrame()

        if "eventName" in original_data.columns:
            results["Event Name"] = original_data["eventName"]

        if "eventTime" in original_data.columns:
            results["Event Time"] = original_data["eventTime"]

        if "eventSource" in original_data.columns:
            results["Event Source"] = original_data["eventSource"]

        results["Prediction"] = [
            "Suspicious" if p == 1 else "Normal"
            for p in predictions
        ]

        if probabilities is not None:
            results["Suspicious Probability (%)"] = (
                probabilities * 100
            ).round(2)

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        total_events = len(predictions)
        suspicious_events = int(np.sum(predictions))
        normal_events = total_events - suspicious_events

        st.subheader("Detection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Events", f"{total_events:,}")
        col2.metric("Suspicious Events", f"{suspicious_events:,}")
        col3.metric("Normal Events", f"{normal_events:,}")

        st.subheader("Prediction Summary")

        st.bar_chart(
            pd.DataFrame(
                {
                    "Events": [normal_events, suspicious_events]
                },
                index=["Normal", "Suspicious"]
            )
        )

        st.subheader("Detailed Detection Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        # -------------------------------------------------
        # DOWNLOAD REPORT
        # -------------------------------------------------

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Detection Report",
            data=csv_data,
            file_name="cloud_security_detection_report.csv",
            mime="text/csv"
        )

        st.success("Cloud activity analysis completed!")

    except Exception as error:

        st.error(f"Error analyzing uploaded file: {error}")
        st.exception(error) 
    
        # -------------------------------------------------
        # PREPROCESS DATA
        # -------------------------------------------------

        with st.spinner("Preprocessing CloudTrail records..."):

            processed_data, original_data = prepare_data(
                records,
                preprocessor,
                preprocessing_info
            )

        # -------------------------------------------------
        # PREDICTIONS
        # -------------------------------------------------
        
                # -------------------------------------------------
        # PREDICTIONS
        # -------------------------------------------------

        with st.spinner("Detecting suspicious activities..."):

            if selected_model == "Random Forest":

                predictions, probabilities = make_predictions(
                    model,
                    processed_data,
                    threshold=model_info.get("threshold", 0.60)
                )

            else:

                ann_input = (
                    processed_data.toarray()
                    if hasattr(processed_data, "toarray")
                    else np.asarray(processed_data)
                )

                ann_input = ann_input.astype("float32")
                ann_input = np.clip(ann_input, -10, 10)

                probabilities = ann_model.predict(
                    ann_input,
                    verbose=0
                ).ravel()

                predictions = (probabilities >= 0.5).astype(int)
                
        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        total_records = len(predictions)

        suspicious_count = int(np.sum(predictions == 1))

        normal_count = int(np.sum(predictions == 0))

        suspicious_percentage = (
            suspicious_count / total_records
        ) * 100
        
# -------------------------------------------------
# ACTIVITY ANALYSIS
# -------------------------------------------------

if "results" in locals() and isinstance(results, pd.DataFrame):

    if "Prediction" in results.columns:

        st.divider()
        st.subheader("📊 Activity Analysis")

        col1, col2 = st.columns(2)

        activity_counts = results["Prediction"].value_counts()

        with col1:
            st.write("Normal vs Suspicious Activities")
            st.bar_chart(activity_counts)

        with col2:
            st.write("Detection Distribution")
            st.dataframe(
                activity_counts.rename("Count"),
                use_container_width=True
            )

        # -------------------------------------------------
        # DISPLAY SUMMARY
        # -------------------------------------------------

        st.divider()

        st.subheader("Detection Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Events", f"{total_records:,}")
        col2.metric("Normal Events", f"{normal_count:,}")
        col3.metric("Suspicious Events", f"{suspicious_count:,}")
        col4.metric("Suspicious Rate", f"{suspicious_percentage:.2f}%")

        # -------------------------------------------------
        # VISUALIZATION
        # -------------------------------------------------

        st.divider()

        st.subheader("Activity Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.write("Normal vs Suspicious Activities")

            chart_data = pd.DataFrame(
                {
                    "Activity": ["Normal", "Suspicious"],
                    "Count": [normal_count, suspicious_count]
                }
            )

            st.bar_chart(
                chart_data.set_index("Activity")
            )

        with chart_col2:

            st.write("Detection Distribution")

            st.write(f"Normal: {normal_count:,}")
            st.progress(normal_count / total_records)

            st.write(f"Suspicious: {suspicious_count:,}")
            st.progress(suspicious_count / total_records)

        # -------------------------------------------------
        # SUSPICIOUS ACTIVITIES
        # -------------------------------------------------

        st.divider()

        st.subheader("🚨 Suspicious Activities")

        suspicious_results = results[
            results["Prediction"] == "Suspicious"
        ]

        if not suspicious_results.empty:

            st.warning(
                f"{len(suspicious_results):,} suspicious events detected!"
            )

            sources = ["All"] + sorted(
                suspicious_results["Event Source"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_source = st.selectbox(
                "Filter by Event Source",
                sources
            )

            if selected_source != "All":
                suspicious_results = suspicious_results[
                    suspicious_results["Event Source"] == selected_source
                ]

            if "Suspicious Probability (%)" in suspicious_results.columns:

                suspicious_results = suspicious_results.sort_values(
                    by="Suspicious Probability (%)",
                    ascending=False
                )

            st.dataframe(
                suspicious_results,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success("No suspicious activities detected.")

       # -------------------------------------------------
# COMPLETE RESULTS
# -------------------------------------------------

if "results" in locals() and isinstance(results, pd.DataFrame):

    st.divider()

    st.subheader("📋 Complete Prediction Results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------
    # DOWNLOAD REPORT
    # -------------------------------------------------

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=csv_data,
        file_name="cloud_security_detection_report.csv",
        mime="text/csv"
    )

    st.success("Cloud activity analysis completed!")

else:
    st.warning("Prediction results are not available.")



# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">

    **Cloud Security Threat Detection Using Machine Learning**

    B.Tech Project | Deep Learning

    Built using Python, Streamlit, Scikit-learn and AWS CloudTrail Dataset.

    </div>
    """,
    unsafe_allow_html=True
)