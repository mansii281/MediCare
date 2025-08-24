import pandas as pd
import numpy as np

np.random.seed(42)

n_low = 500
n_high = 500

# ------------------- Low Risk Data -------------------
low_risk = pd.DataFrame({
    "Age": np.random.randint(20, 50, n_low),
    "BMI": np.round(np.random.uniform(18, 24, n_low), 1),
    "BloodPressure": np.random.randint(90, 120, n_low),
    "Cholesterol": np.random.randint(150, 200, n_low),
    "Glucose": np.random.randint(70, 110, n_low),
    "HeartRate": np.random.randint(60, 80, n_low),
    "Target": 0
})

# ------------------- High Risk Data -------------------
high_risk = pd.DataFrame({
    "Age": np.random.randint(45, 80, n_high),
    "BMI": np.round(np.random.uniform(25, 40, n_high), 1),
    "BloodPressure": np.random.randint(130, 180, n_high),
    "Cholesterol": np.random.randint(210, 300, n_high),
    "Glucose": np.random.randint(110, 200, n_high),
    "HeartRate": np.random.randint(75, 120, n_high),
    "Target": 1
})

# ------------------- Combine & Shuffle -------------------
data = pd.concat([low_risk, high_risk], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data.to_csv("dataset.csv", index=False)
print("✅ Synthetic dataset saved as dataset.csv")
