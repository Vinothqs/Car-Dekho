import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("car_price_model.pkl")

# Load feature names
feature_names = model.feature_names_in_

def preprocess_input(year, kms_driven, city, fuel_type, owner_type, body_type):
    """Convert user inputs into the correct format for prediction."""
    # Create a DataFrame with all feature columns set to 0
    data = pd.DataFrame(columns=feature_names)
    data.loc[0] = [0] * len(feature_names)
    
    # Assign user input values
    data["Year"] = year
    data["Kms_Driven"] = kms_driven
    
    # One-hot encoding for categorical features
    if f"City_{city}" in data.columns:
        data[f"City_{city}"] = 1
    if f"Fuel Type_{fuel_type}" in data.columns:
        data[f"Fuel Type_{fuel_type}"] = 1
    if f"Owner Type_{owner_type}" in data.columns:
        data[f"Owner Type_{owner_type}"] = 1
    if f"Body Type_{body_type}" in data.columns:
        data[f"Body Type_{body_type}"] = 1
    
    return data

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter car details to predict the estimated price.")

# User Inputs
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
city = st.selectbox("City", ["Chennai", "Bangalore", "Delhi", "Hyderabad", "Jaipur", "Kolkata"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])
owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
body_type = st.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Wagon"])

# Predict button
if st.button("Predict Price"):
    # Preprocess input
    input_data = preprocess_input(year, kms_driven, city, fuel_type, owner_type, body_type)
    
    # Make prediction
    price_pred = model.predict(input_data)[0]
    
    # Display result
    st.success(f"Estimated Car Price: ₹{price_pred:,.2f}")