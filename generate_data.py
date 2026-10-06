import numpy as np
import pandas as pd
import datetime

rng = np.random.default_rng(42)

N = 50000  # number of sessions

# ---------- Dimension distributions ----------
devices = rng.choice(
    ["Mobile", "Desktop", "Tablet"],
    size=N,
    p=[0.58, 0.32, 0.10]
)

traffic_sources = rng.choice(
    ["Organic Search", "Paid Search", "Social Media", "Direct", "Email"],
    size=N,
    p=[0.30, 0.24, 0.20, 0.16, 0.10]
)

categories = rng.choice(
    ["Electronics", "Fashion", "Home & Kitchen", "Beauty & Personal Care", "Sports & Outdoors"],
    size=N,
    p=[0.26, 0.28, 0.18, 0.16, 0.12]
)

user_type = rng.choice(
    ["New", "Returning"],
    size=N,
    p=[0.68, 0.32]
)

# session dates over a 90-day window
start_date = datetime.date(2025, 1, 1)
session_dates = [start_date + datetime.timedelta(days=int(d)) for d in rng.integers(0, 90, size=N)]

user_ids = [f"U{100000+i}" for i in rng.integers(0, 38000, size=N)]  # some repeat = returning-ish pool
session_ids = [f"S{200000+i}" for i in range(N)]

df = pd.DataFrame({
    "User_ID": user_ids,
    "Session_ID": session_ids,
    "Session_Date": session_dates,
    "Device": devices,
    "Traffic_Source": traffic_sources,
    "Product_Category": categories,
    "User_Type": user_type,
})

# ---------- Funnel simulation ----------
# Stage 1: Homepage visit -- every session in this table starts here
df["Homepage_Visit"] = 1

# Stage 2: Search
base_search_p = 0.72
traffic_search_adj = df["Traffic_Source"].map({
    "Organic Search": 0.06,
    "Paid Search": 0.10,
    "Social Media": -0.10,
    "Direct": -0.14,
    "Email": 0.02,
})
returning_adj = np.where(df["User_Type"] == "Returning", 0.06, 0.0)
search_p = np.clip(base_search_p + traffic_search_adj.values + returning_adj, 0.05, 0.97)
df["Search"] = (rng.random(N) < search_p).astype(int)

# Stage 3: Product View (only possible if searched)
base_pv_p = 0.66
device_pv_adj = df["Device"].map({"Desktop": 0.03, "Mobile": -0.02, "Tablet": 0.00})
pv_p = np.clip(base_pv_p + device_pv_adj.values + returning_adj, 0.05, 0.95)
df["Product_View"] = np.where(df["Search"] == 1, (rng.random(N) < pv_p).astype(int), 0)

# Stage 4: Add to Cart (only if viewed a product)
# This is where we deliberately encode a strong, discoverable mobile drop-off
base_cart_p = 0.50
device_cart_adj = df["Device"].map({"Desktop": 0.12, "Mobile": -0.22, "Tablet": -0.04})
returning_cart_adj = np.where(df["User_Type"] == "Returning", 0.08, 0.0)
cart_p = np.clip(base_cart_p + device_cart_adj.values + returning_cart_adj, 0.05, 0.95)
df["Add_to_Cart"] = np.where(df["Product_View"] == 1, (rng.random(N) < cart_p).astype(int), 0)

# Stage 5: Checkout (only if added to cart)
base_checkout_p = 0.66
device_checkout_adj = df["Device"].map({"Desktop": 0.04, "Mobile": -0.06, "Tablet": 0.00})
checkout_p = np.clip(base_checkout_p + device_checkout_adj.values + returning_cart_adj, 0.05, 0.95)
df["Checkout"] = np.where(df["Add_to_Cart"] == 1, (rng.random(N) < checkout_p).astype(int), 0)

# Stage 6: Purchase (only if reached checkout)
base_purchase_p = 0.63
traffic_purchase_adj = df["Traffic_Source"].map({
    "Organic Search": 0.03,
    "Paid Search": 0.02,
    "Social Media": -0.05,
    "Direct": 0.05,
    "Email": 0.06,
})
purchase_p = np.clip(base_purchase_p + traffic_purchase_adj.values + returning_cart_adj, 0.05, 0.97)
df["Purchase"] = np.where(df["Checkout"] == 1, (rng.random(N) < purchase_p).astype(int), 0)

# ---------- Order value (only for purchases) ----------
category_base_value = {
    "Electronics": 145,
    "Fashion": 62,
    "Home & Kitchen": 78,
    "Beauty & Personal Care": 38,
    "Sports & Outdoors": 70,
}
base_vals = df["Product_Category"].map(category_base_value).values
noise = rng.normal(1.0, 0.35, N)
order_value = np.round(np.clip(base_vals * noise, 8, None), 2)
df["Order_Value"] = np.where(df["Purchase"] == 1, order_value, 0)

# ---------- Cleanup / column order ----------
df["Session_Date"] = pd.to_datetime(df["Session_Date"]).dt.strftime("%Y-%m-%d")

col_order = [
    "User_ID", "Session_ID", "Session_Date", "Device", "Traffic_Source",
    "Product_Category", "User_Type", "Homepage_Visit", "Search",
    "Product_View", "Add_to_Cart", "Checkout", "Purchase", "Order_Value"
]
df = df[col_order]

df.to_csv("/home/claude/proj/ecommerce_funnel_data.csv", index=False)

# ---------- Quick sanity summary printed to console ----------
funnel_cols = ["Homepage_Visit", "Search", "Product_View", "Add_to_Cart", "Checkout", "Purchase"]
print("Overall funnel:")
print(df[funnel_cols].sum())
print()
print("Add_to_Cart rate by device (Product_View -> Add_to_Cart):")
pv = df[df["Product_View"] == 1]
print(pv.groupby("Device")["Add_to_Cart"].mean().round(3))
print()
print("Rows:", len(df))
print("Total revenue:", df["Order_Value"].sum().round(2))
