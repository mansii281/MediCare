# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Load Dataset ----------------
data = pd.read_csv("/Users/mansisharma/medicare/dataset.csv")  # Updated path

# Features and target
X = data.drop("Target", axis=1)
y = data["Target"]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Train Random Forest ----------------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------- Train XGBoost ----------------
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)

# ---------------- Evaluate Models ----------------
def evaluate_model(model, X, y, dataset_name, model_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n📊 {model_name} Evaluation on {dataset_name} Set:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low Risk (0)", "High Risk (1)"],
                yticklabels=["Low Risk (0)", "High Risk (1)"])
    plt.title(f"{model_name} - Confusion Matrix ({dataset_name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

# ---------------- Evaluate on Train Set ----------------
evaluate_model(rf_model, X_train, y_train, "Train", "Random Forest")
evaluate_model(xgb_model, X_train, y_train, "Train", "XGBoost")

# ---------------- Evaluate on Test Set ----------------
evaluate_model(rf_model, X_test, y_test, "Test", "Random Forest")
evaluate_model(xgb_model, X_test, y_test, "Test", "XGBoost")

# ---------------- Save Models ----------------
joblib.dump(rf_model, "rf_cls.pkl")
joblib.dump(xgb_model, "xgb_cls.pkl")

print("\n✅ Models saved as rf_cls.pkl and xgb_cls.pkl")
