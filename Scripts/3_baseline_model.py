# Import libraries
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


# Create output folder
os.makedirs("Outputs", exist_ok=True)

# Load feature engineered dataset
data = pd.read_csv("Outputs/feature_engineered_data.csv")

# Remove date column (model cannot use raw dates)
data = data.drop(columns=["sales_date"])

# One-hot encode categorical features
data = pd.get_dummies(data, columns=["store_type"], drop_first=True)

# Define target
y = data["weekly_revenue"]

# Define features
X = data.drop(columns=["weekly_revenue"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start timer
start_time = time.time()

# Train RandomForest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# End timer
runtime = time.time() - start_time

# Evaluation metrics
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)


# Throughput
throughput = len(X_test) / runtime

# Save results
results = pd.DataFrame({
    "model": ["RandomForest"],
    "RMSE": [rmse],
    "MAE": [mae],
    "runtime_seconds": [runtime],
    "throughput_rows_per_sec": [throughput]
})

results.to_csv("Outputs/baseline_results.csv", index=False)

print("Baseline model results saved.")