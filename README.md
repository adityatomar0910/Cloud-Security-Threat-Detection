# 🔐 Cloud Security Threat Detection Using Machine Learning and Deep Learning

An intelligent cloud security monitoring system that analyzes AWS CloudTrail activity logs and detects potentially suspicious activities using Machine Learning and Deep Learning techniques.

---

## 📌 Project Overview

Cloud environments generate large volumes of activity logs. Identifying suspicious activities manually can be time-consuming and difficult.

This project aims to detect potentially malicious or abnormal cloud activities by analyzing AWS CloudTrail logs using trained Machine Learning and Deep Learning models.

The system provides an interactive Streamlit dashboard where users can upload CloudTrail JSON files, analyze cloud activities, view detection results, and download reports.

## 🎯 Project Objectives

- Analyze AWS CloudTrail activity logs.
- Identify normal and potentially suspicious activities.
- Apply Machine Learning and Deep Learning techniques for anomaly detection.
- Compare Random Forest and Artificial Neural Network (ANN) models.
- Provide an interactive security analysis dashboard.
- Generate downloadable detection reports.

## ✨ Key Features

- **CloudTrail Log Analysis:** Upload AWS CloudTrail JSON files for analysis.
- **Machine Learning Detection:** Uses an improved Random Forest classifier.
- **Deep Learning Detection:** Uses an Artificial Neural Network (ANN).
- **Model Selection:** Choose between Random Forest and ANN.
- **Automated Predictions:** Classifies events as Normal or Suspicious.
- **Detection Probability:** Displays suspicious activity probability where available.
- **Interactive Dashboard:** Visualizes detection statistics and results.
- **CSV Report Generation:** Download prediction results for further analysis.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Streamlit |
| Model Serialization | Joblib |
| Dataset Format | JSON |
| Development Environment | Jupyter Notebook, VS Code, Ubuntu WSL |

## 🧠 Models Implemented

### 1. Random Forest

- An ensemble-based Machine Learning classification algorithm.
- Used to classify cloud activity logs into Normal and Suspicious categories.
- Uses a detection threshold of 0.60 in the current dashboard configuration.

### 2. Artificial Neural Network (ANN)

- A Deep Learning model trained to identify suspicious cloud activities.
- Uses TensorFlow and Keras.
- Produces prediction probabilities for uploaded cloud activity records.
- Uses a classification threshold of 0.50 in the current dashboard configuration.

## 📊 Model Performance

The following results were obtained during evaluation on the held-out test dataset.

| Evaluation Metric | Random Forest | ANN |
|---|---:|---:|
| Accuracy | 91.80% | 74.08% |
| Precision | 96
