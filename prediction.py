import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Historical 6-month data
historical = {
    'month': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'],
    'order_quantity': [120, 135, 150, 160, 155, 170],
    'price': [50, 52, 51, 53, 55, 54],
    'inventory': [300, 280, 290, 310, 300, 295],
    'safety_stock': [70, 70, 75, 78, 76, 80]
}
df = pd.DataFrame(historical)
df['month_num'] = np.arange(1, len(df) + 1)

# Forecast next 3 months
future_months = [7, 8, 9]
future_labels = ['2024-07', '2024-08', '2024-09']
model = LinearRegression()
model.fit(df[['month_num']], df[['order_quantity']])
forecasted_demand = model.predict(np.array(future_months).reshape(-1, 1)).flatten().round().astype(int)

# Seller data
seller_data = {
    '2024-07': [('SellerA', 44, 2, 200), ('SellerB', 45, 3, 150), ('SellerC', 43, 5, 90)],
    '2024-08': [('SellerA', 46, 5, 80), ('SellerB', 45, 6, 100), ('SellerC', 47, 3, 150)],
    '2024-09': [('SellerA', 48, 6, 90), ('SellerB', 46, 7, 70), ('SellerC', 45, 4, 95)],
}

# Parameters
safety_stock_min = 60
inventory = df['inventory'].iloc[-1]

# Decision logic
decision_output = []

for i, month in enumerate(future_labels):
    demand = forecasted_demand[i]
    available_inventory = inventory
    remaining_inventory = available_inventory - demand

    # Determine how much to add to meet exact safety stock
    extra_needed = max(safety_stock_min - remaining_inventory, 0)
    total_required = demand + extra_needed

    sellers = seller_data[month]

    # Strict filter: lead time ≤ 4 and enough quantity
    strict = [s for s in sellers if s[2] <= 4 and s[3] >= total_required]
    if strict:
        selected = min(strict, key=lambda x: x[1] * total_required)
        remark = "✔ Strict seller"
    else:
        relaxed = [s for s in sellers if s[2] <= 6 and s[3] >= total_required]
        if relaxed:
            selected = min(relaxed, key=lambda x: x[1] * total_required)
            remark = "⚠ Relaxed lead time"
        else:
            selected = max(sellers, key=lambda x: x[3])  # partial supplier
            total_required = selected[3]
            extra_needed = max(total_required - demand, 0)
            remark = f"‼ Partial supply used"

    name, unit_price, lead_time, available_qty = selected
    total_cost = unit_price * total_required

    # Set next month's inventory to exactly safety stock min
    inventory = safety_stock_min

    decision_output.append((
        month, name, unit_price, demand, extra_needed, total_required,
        total_cost, safety_stock_min, remark
    ))

# Output table
df_result = pd.DataFrame(decision_output, columns=[
    "Month", "Seller", "Unit Price", "Demand", "Added to Safety Stock",
    "Ordered Qty", "Total Cost", "Ending Safety Stock", "Remarks"
])
print("\n🔎 Final Plan with Fixed Safety Stock = 60 Every Month:\n")
print(df_result.to_string(index=False))