# 🩺 MediTrack: Smart Health Risk Detector

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🌟 Overview

**MediTrack** is an AI-powered health risk detection and tracking system. It predicts whether a patient is at **Low Risk** ✅ or **High Risk** 🚨 based on vital health metrics.

* Uses **Random Forest** and **XGBoost** classifiers.
* Supports **hybrid averaging** for improved accuracy.
* Provides **explainable insights** via **SHAP**.
* Generates **interactive dashboards** and **PDF medical reports**.

---

## 🧾 Input Features

| Feature       | Type    | Description                    |
| ------------- | ------- | ------------------------------ |
| Age           | Integer | Patient age in years           |
| BMI           | Float   | Body Mass Index                |
| BloodPressure | Integer | Systolic blood pressure (mmHg) |
| Cholesterol   | Integer | Cholesterol level (mg/dL)      |
| Glucose       | Integer | Glucose level (mg/dL)          |
| HeartRate     | Integer | Heart rate (bpm)               |
| Doctor Notes  | Text    | Optional notes by doctor       |

---

## 📊 Predicted Output

* **Low Risk** ✅ – Patient is healthy; maintain lifestyle.
* **High Risk** 🚨 – Patient is at high risk; consult a doctor immediately.

Predictions are computed from **Random Forest** and **XGBoost**, optionally averaged for a **hybrid model**.

---

## 🔬 SHAP Feature Importance

* SHAP explains how **each feature contributes** to the risk prediction.
* Highlights **most influential health metrics** for clinicians.

---

## 📈 Model Evaluation Metrics

### Random Forest – Train Set

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 1.0000 |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |

**Classification Report**

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       400
           1       1.00      1.00      1.00       400
    accuracy                           1.00       800
   macro avg       1.00      1.00      1.00       800
weighted avg       1.00      1.00      1.00       800
```

### XGBoost – Train Set

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 1.0000 |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |

**Classification Report**

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       400
           1       1.00      1.00      1.00       400
    accuracy                           1.00       800
   macro avg       1.00      1.00      1.00       800
weighted avg       1.00      1.00      1.00       800
```

### Random Forest – Test Set

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 1.0000 |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |

**Classification Report**

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       100
           1       1.00      1.00      1.00       100
    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```

### XGBoost – Test Set

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 1.0000 |
| Precision | 1.0000 |
| Recall    | 1.0000 |
| F1-Score  | 1.0000 |

**Classification Report**

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       100
           1       1.00      1.00      1.00       100
    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```

---

## 💾 Saved Models

* **Random Forest:** `rf_cls.pkl`
* **XGBoost:** `xgb_cls.pkl`
* **Hybrid Model:** Combination of RF and XGBoost predictions

---

## 🛠️ Libraries Used

* `streamlit` – Interactive web interface
* `pandas`, `numpy` – Data handling
* `scikit-learn` – Random Forest, metrics
* `xgboost` – XGBoost classifier
* `shap` – Feature importance visualization
* `plotly` – Interactive charts
* `reportlab` – PDF report generation

---

## 📁 Project Structure

```
MediTrack/
├── app.py                # Streamlit app
├── rf_cls.pkl            # Random Forest model
├── xgb_cls.pkl           # XGBoost model
├── dataset.csv           # Sample dataset
├── README.md             # Project documentation
└── requirements.txt      # Required libraries
```

---

## 🚀 How to Run

1. Clone the repository.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Enter patient health information to view **risk assessment**, **SHAP feature importance**, and **download PDF report**.

---

## 🔗 References

* [SHAP Documentation](https://shap.readthedocs.io/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)

---
