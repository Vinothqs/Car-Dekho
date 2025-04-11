import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load the cleaned dataset
df = pd.read_csv("Cleaned_Car_Dekho.csv")

# Function to extract numeric price
def extract_price(row):
    try:
        details = json.loads(row) if isinstance(row, str) else {}
        price_str = details.get("price", "").replace("?", "").strip()
        
        # Extract numeric part
        price_match = re.search(r"(\d+(\.\d+)?)", price_str)
        if price_match:
            price_value = float(price_match.group(1))
            if "Lakh" in price_str:
                return price_value * 100000  # Convert Lakh to numeric
            elif "Cr" in price_str:
                return price_value * 10000000  # Convert Crore to numeric
            else:
                return price_value  # Assume it's already in INR
        return np.nan
    except:
        return np.nan

# Apply the function to extract price
df["priceActual"] = df["new_car_detail"].apply(extract_price)

# Drop rows where price is missing
df = df.dropna(subset=["priceActual"])

# Convert 'Kms_Driven' to numeric (remove commas and convert to integer)
df["Kms_Driven"] = df["Kms_Driven"].astype(str).str.replace(",", "").astype(float)

# Fill missing values
df["Kms_Driven"].fillna(df["Kms_Driven"].median(), inplace=True)

# Drop unnecessary columns
df = df.drop(columns=["new_car_detail", "new_car_overview", "new_car_feature", "new_car_specs", "car_links"])

# Separate features (X) and target variable (y)
X = df.drop(columns=["priceActual"])  # Features
y = df["priceActual"]  # Target variable

# Convert categorical features to numerical
X = pd.get_dummies(X, drop_first=True)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the trained model
joblib.dump(model, "car_price_model.pkl")
print("Model saved successfully!")