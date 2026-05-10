# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create folders
os.makedirs("Outputs", exist_ok=True)
os.makedirs("Outputs/figures", exist_ok=True)

# Load feature engineered data
data = pd.read_csv("Outputs/feature_engineered_data.csv")

# -----------------------------
# Calculate Total Promotions
# -----------------------------
data["Total_Promo"] = (
    data["holiday_discount"] +
    data["clearance_discount"] +
    data["seasonal_discount"] +
    data["weekend_discount"] +
    data["special_event_discount"]
)

# Identify high promotion weeks
data["High_Promo"] = data["Total_Promo"] > data["Total_Promo"].median()

# -----------------------------
# Calculate averages
# -----------------------------
promo_sales = data[data["High_Promo"] == True]["weekly_revenue"].mean()
nonpromo_sales = data[data["High_Promo"] == False]["weekly_revenue"].mean()

cannibalization = promo_sales - nonpromo_sales

# Save results
results = pd.DataFrame({
    "Promo_Sales_Avg": [promo_sales],
    "NonPromo_Sales_Avg": [nonpromo_sales],
    "Cannibalization_Effect": [cannibalization]
})

results.to_csv("Outputs/cannibalization_results.csv", index=False)
print("Cannibalization analysis done successfully")
