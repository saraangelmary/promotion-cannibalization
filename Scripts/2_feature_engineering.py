# Import pandas
import pandas as pd

# Load cleaned dataset from Script 1
data = pd.read_csv("Outputs/clean_data.csv")

# Convert sales_date column to datetime
data["sales_date"] = pd.to_datetime(data["sales_date"])

# -------------------------
# TIME FEATURES
# -------------------------

# Extract year
data["Year"] = data["sales_date"].dt.year

# Extract month
data["Month"] = data["sales_date"].dt.month

# Extract week number
data["Week"] = data["sales_date"].dt.isocalendar().week.astype(int)

# Extract day of month
data["Day"] = data["sales_date"].dt.day

# Extract day of week
data["DayOfWeek"] = data["sales_date"].dt.dayofweek

# Weekend indicator
data["IsWeekend"] = data["DayOfWeek"].isin([5,6]).astype(int)

# -------------------------
# SORT DATA
# -------------------------

data = data.sort_values(["store_id","department_id","sales_date"])

# -------------------------
# LAG FEATURES
# -------------------------

data["Sales_Last_Week"] = data.groupby(
["store_id","department_id"]
)["weekly_revenue"].shift(1)

data["Sales_Last_2_Weeks"] = data.groupby(
["store_id","department_id"]
)["weekly_revenue"].shift(2)

data["Sales_Last_4_Weeks"] = data.groupby(
["store_id","department_id"]
)["weekly_revenue"].shift(4)

data["Sales_Last_12_Weeks"] = data.groupby(
["store_id","department_id"]
)["weekly_revenue"].shift(12)

# -------------------------
# ROLLING FEATURES (SAFE VERSION)
# -------------------------

data["Rolling_4_Week_Sales"] = data.groupby(
["store_id","department_id"]
)["weekly_revenue"].transform(lambda x: x.rolling(4).mean())

data["Rolling_12_Week_Sales"] = data.groupby(
["store_id","department_id"]
)["weekly_revenue"].transform(lambda x: x.rolling(12).mean())

# -------------------------
# PROMOTION TOTAL FEATURE
# -------------------------

data["Promo_Total"] = data[
[
"holiday_discount",
"clearance_discount",
"seasonal_discount",
"weekend_discount",
"special_event_discount"
]
].sum(axis=1)

# -------------------------
# REMOVE NA VALUES
# -------------------------

data = data.dropna()

# -------------------------
# SAVE OUTPUT DATASET
# -------------------------

data.to_csv("Outputs/feature_engineered_data.csv", index=False)

print("Feature engineering completed successfully.")

