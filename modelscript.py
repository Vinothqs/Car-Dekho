import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the cleaned dataset
df = pd.read_csv("Cleaned_Car_Dekho.csv")

# Function to extract numeric price
def extract_price(row):
    try:
        details = json.loads(row) if isinstance(row, str) else {}
        price_str = details.get("price", "").replace("?", "").strip()
        price_match = re.search(r"(\d+(\.\d+)?)", price_str)
        if price_match:
            price_value = float(price_match.group(1))
            if "Lakh" in price_str:
                return price_value * 100000
            elif "Cr" in price_str:
                return price_value * 10000000
            else:
                return price_value
        return np.nan
    except:
        return np.nan

# Apply function to extract price
df["priceActual"] = df["new_car_detail"].apply(extract_price)

# Drop rows with missing price
df.dropna(subset=["priceActual"], inplace=True)

# Convert Kms_Driven to numeric
df["Kms_Driven"] = df["Kms_Driven"].astype(str).str.replace(",", "").astype(float)
df["Kms_Driven"].fillna(df["Kms_Driven"].median(), inplace=True)

# Drop unnecessary columns
df.drop(columns=["new_car_detail", "new_car_overview", "new_car_feature", "new_car_specs", "car_links"], inplace=True)

# Split into features and target
X = df.drop(columns=["priceActual"])
y = df["priceActual"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
}

# Train, predict, evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"model": model, "mae": mae}
    print(f"{name} MAE: ₹{mae:,.2f}")

# Select best model with MAE < 2 Lakhs
best_model_name = None
best_mae = float("inf")

for name, result in results.items():
    if result["mae"] < best_mae and result["mae"] < 200000:
        best_mae = result["mae"]
        best_model_name = name

# Save best model
if best_model_name:
    joblib.dump(results[best_model_name]["model"], "car_price_model.pkl")
    print(f"\n✅ Best model: {best_model_name} with MAE ₹{best_mae:,.2f} saved as car_price_model.pkl")
else:
    print("\n❌ No model achieved MAE below ₹2,00,000. Try tuning hyperparameters.")