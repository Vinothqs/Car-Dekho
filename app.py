import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("car_price_model.pkl")
feature_names = model.feature_names_in_

# Load dataset to extract dropdown options
@st.cache_data
def load_dropdown_data():
    df = pd.read_csv("Cleaned_Car_Dekho.csv")
    brands = sorted(df["Brand"].dropna().unique())
    models = sorted(df["Model"].dropna().unique())
    return brands, models

brands, models = load_dropdown_data()

# Function to preprocess input
def preprocess_input(year, kms_driven, city, fuel_type, owner_type, body_type, brand, model_name):
    data = pd.DataFrame(columns=feature_names)
    data.loc[0] = [0] * len(feature_names)

    # Assign numeric fields
    data["Year"] = year
    data["Kms_Driven"] = kms_driven

    # One-hot encode categorical fields if present in model
    for col_name, value in {
        f"City_{city}": city,
        f"Fuel Type_{fuel_type}": fuel_type,
        f"Owner Type_{owner_type}": owner_type,
        f"Body Type_{body_type}": body_type,
        f"Brand_{brand}": brand,
        f"Model_{model_name}": model_name
    }.items():
        if col_name in data.columns:
            data[col_name] = 1

    return data

# UI
st.title("🚗 Car Price Prediction App")
st.markdown("Enter the car details below to get an estimated selling price.")

# Input widgets
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500)

city = st.selectbox("City", ["Chennai", "Bangalore", "Delhi", "Hyderabad", "Jaipur", "Kolkata"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
body_type = st.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Wagon"])

brand = st.selectbox("Car Brand", options=brands)
model_name = st.selectbox("Car Model", options=models)

# Predict button
if st.button("Predict Price"):
    input_df = preprocess_input(year, kms_driven, city, fuel_type, owner_type, body_type, brand, model_name)
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: ₹ {prediction:,.2f}")