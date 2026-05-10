# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create figures folder if it doesn't exist
os.makedirs("Outputs/figures", exist_ok=True)

# -----------------------------
# 1️⃣ Model Comparison Charts
# -----------------------------
model_metrics = pd.read_csv("Outputs/model_comparison.csv")

# RMSE
plt.figure()
plt.bar(model_metrics["model"], model_metrics["RMSE"])
plt.ylabel("RMSE")
plt.title("RMSE Comparison")
plt.savefig("Outputs/figures/rmse_comparison.png")
plt.close()

# MAE
plt.figure()
plt.bar(model_metrics["model"], model_metrics["MAE"])
plt.ylabel("MAE")
plt.title("MAE Comparison")
plt.savefig("Outputs/figures/mae_comparison.png")
plt.close()

# Runtime
plt.figure()
plt.bar(model_metrics["model"], model_metrics["runtime_seconds"])
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison")
plt.savefig("Outputs/figures/runtime_comparison.png")
plt.close()

# Throughput
plt.figure()
plt.bar(model_metrics["model"], model_metrics["throughput_rows_per_sec"])
plt.ylabel("Throughput (rows/sec)")
plt.title("Throughput Comparison")
plt.savefig("Outputs/figures/throughput_comparison.png")
plt.close()

# -----------------------------
# 2️⃣ Simple Data Analysis Charts
# -----------------------------
data = pd.read_csv("Outputs/feature_engineered_data.csv")

# Monthly Sales Trend
monthly_sales = data.groupby("Month")["weekly_revenue"].mean()
plt.figure()
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.xlabel("Month")
plt.ylabel("Average Weekly Revenue (INR)")
plt.title("Monthly Sales Trend")
plt.savefig("Outputs/figures/monthly_sales_trend.png")
plt.close()

# Top 10 Departments by Sales
dept_sales = data.groupby("department_id")["weekly_revenue"].mean().sort_values(ascending=False).head(10)
plt.figure()
plt.bar(dept_sales.index.astype(str), dept_sales.values)
plt.xlabel("Department ID")
plt.ylabel("Average Weekly Revenue (INR)")
plt.title("Top 10 Departments by Sales")
plt.savefig("Outputs/figures/top_departments_sales.png")
plt.close()

# Promotion Discount vs Sales
data["Total_Promo"] = (
    data["holiday_discount"] +
    data["clearance_discount"] +
    data["seasonal_discount"] +
    data["weekend_discount"] +
    data["special_event_discount"]
)
plt.figure()
plt.scatter(data["Total_Promo"], data["weekly_revenue"])
plt.xlabel("Total Promotion Discount (INR)")
plt.ylabel("Weekly Revenue (INR)")
plt.title("Promotion Discount vs Sales")
plt.savefig("Outputs/figures/promo_discount_vs_sales.png")
plt.close()

# -----------------------------
# 3️⃣ Cannibalization Charts
# -----------------------------
cannibal_data = pd.read_csv("Outputs/cannibalization_results.csv")

# Promo vs Non-Promo Sales
sales_types = ["Promo Sales Avg", "Non-Promo Sales Avg"]
sales_values = [cannibal_data["Promo_Sales_Avg"][0], cannibal_data["NonPromo_Sales_Avg"][0]]

plt.figure(figsize=(6,5))
plt.bar(sales_types, sales_values, color=['green','orange'])
plt.ylabel("Average Weekly Sales (INR)")
plt.title("Promotion vs Non-Promotion Sales")
plt.savefig("Outputs/figures/promo_vs_nonpromo.png")
plt.close()

# Cannibalization Effect
plt.figure(figsize=(5,4))
plt.bar(["Cannibalization Effect"], [cannibal_data["Cannibalization_Effect"][0]], color='red')
plt.ylabel("Sales Lost Due to Cannibalization (INR)")
plt.title("Promotional Cannibalization Effect")
plt.savefig("Outputs/figures/cannibalization_effect.png")
plt.close()

print("All charts (including cannibalization) generated and saved in Outputs/figures/")