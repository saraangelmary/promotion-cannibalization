# Import pandas for data processing
import pandas as pd

# Import os to manage folders
import os

# Create Outputs directory if it does not exist
os.makedirs("Outputs", exist_ok=True)

# Load datasets
train = pd.read_csv("Data/train.csv")
features = pd.read_csv("Data/features.csv")
stores = pd.read_csv("Data/stores.csv")

# Merge sales data with feature data
data = pd.merge(train, features, on=["Store", "Date"], how="left")

# Merge with store metadata
data = pd.merge(data, stores, on="Store", how="left")

# Rename columns to business-friendly names
data.rename(columns={
    "Store": "store_id",
    "Dept": "department_id",
    "Date": "sales_date",
    "Weekly_Sales": "weekly_revenue",
    "IsHoliday": "holiday_flag",
    "Temperature": "store_temperature",
    "Fuel_Price": "fuel_price_index",
    "CPI": "consumer_price_index",
    "Unemployment": "unemployment_rate",
    "Type": "store_type",
    "Size": "store_size_sqft",
    "MarkDown1": "holiday_discount",
    "MarkDown2": "clearance_discount",
    "MarkDown3": "seasonal_discount",
    "MarkDown4": "weekend_discount",
    "MarkDown5": "special_event_discount"
}, inplace=True)

# Promotion columns
promo_cols = [
    "holiday_discount",
    "clearance_discount",
    "seasonal_discount",
    "weekend_discount",
    "special_event_discount"
]

# Replace missing promotion values with 0
data[promo_cols] = data[promo_cols].fillna(0)

# USD → INR conversion rate
usd_to_inr = 83

# Convert weekly revenue to INR
data["weekly_revenue"] = (data["weekly_revenue"] * usd_to_inr).round(0).astype(int)

# Convert all promotion values to INR
data[promo_cols] = (data[promo_cols] * usd_to_inr).round(0).astype(int)

# Save cleaned dataset
data.to_csv("Outputs/clean_data.csv", index=False)

print("Clean dataset saved with INR values.")