# Import libraries
import pandas as pd
import time
import os
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Create output folder
os.makedirs("Outputs", exist_ok=True)

# Load dataset
data = pd.read_csv("Outputs/feature_engineered_data.csv")

# Remove date
data = data.drop(columns=["sales_date"])

# One-hot encode categorical
data = pd.get_dummies(data, columns=["store_type"], drop_first=True)

# Target
y = data["weekly_revenue"]

# Features
X = data.drop(columns=["weekly_revenue"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start timer
start_time = time.time()

# LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "verbose": -1
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=300
)

# Predictions
predictions = model.predict(X_test)

# End timer
runtime = time.time() - start_time

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
throughput = len(X_test) / runtime

# Save LightGBM results
results = pd.DataFrame({
    "model": ["LightGBM"],
    "RMSE": [rmse],
    "MAE": [mae],
    "runtime_seconds": [runtime],
    "throughput_rows_per_sec": [throughput]
})
results.to_csv("Outputs/lightgbm_results.csv", index=False)
print("LightGBM results saved.")

# Save predictions for charts (Actual vs Predicted)
pred_df = pd.DataFrame({
    "actual_sales": y_test.values,
    "predicted_sales": predictions
})
pred_df.to_csv("Outputs/lgbm_preds.csv", index=False)
print("Predictions saved as lgbm_preds.csv")

# Combine with baseline results
baseline_file = "Outputs/baseline_results.csv"
if os.path.exists(baseline_file):
    baseline = pd.read_csv(baseline_file)
    model_comparison = pd.concat([baseline, results], ignore_index=True)
    model_comparison.to_csv("Outputs/model_comparison.csv", index=False)
    print("Model comparison saved as model_comparison.csv")
else:
    print("Baseline results not found. Only LightGBM results saved.")