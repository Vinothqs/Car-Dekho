import streamlit as st
import pandas as pd
import json
import re  # Import regex for cleaning data

def clean_price(price):
    """Extract numeric value from price string (₹1.5 Lakh -> 150000)"""
    if isinstance(price, str):
        price = price.replace("₹", "").replace(",", "").strip()
        if "Lakh" in price:
            price = re.sub(r"[^\d.]", "", price)  # Remove non-numeric characters
            try:
                return int(float(price) * 100000)  # Convert Lakh to actual number
            except ValueError:
                return None
        elif price.isdigit():
            return int(price)  # Convert normal numbers directly
    return None  # If price format is invalid, return None

def clean_kilometers(kms):
    """Convert kilometer strings ('99,929' -> 99929) into integers"""
    if isinstance(kms, str):
        kms = kms.replace(",", "").strip()  # Remove commas
        if kms.isdigit():
            return int(kms)  # Convert to integer
    return None  # Return None if invalid

def load_data():
    df = pd.read_csv("Cleaned_Car_Dekho.csv")
    
    # Convert JSON-like columns (if they contain JSON data)
    json_columns = ['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs']
    for col in json_columns:
        df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {})

    # Extract important details from 'new_car_detail'
    df['Brand'] = df['new_car_detail'].apply(lambda x: x.get('oem', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['Model'] = df['new_car_detail'].apply(lambda x: x.get('model', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['Variant'] = df['new_car_detail'].apply(lambda x: x.get('variantName', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['Price'] = df['new_car_detail'].apply(lambda x: x.get('price', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['Price'] = df['Price'].apply(clean_price)  # Convert price to integer
    df['Fuel Type'] = df['Fuel_Type']
    df['Transmission'] = df['new_car_detail'].apply(lambda x: x.get('transmission', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['Owner'] = df['new_car_detail'].apply(lambda x: x.get('owner', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['Body Type'] = df['new_car_detail'].apply(lambda x: x.get('bt', 'Unknown') if isinstance(x, dict) else 'Unknown')
    df['Kilometers'] = df['Kms_Driven'].apply(clean_kilometers)  # Convert kilometers to integer
    df['City'] = df['City']

    return df.dropna(subset=['Price', 'Kilometers'])  # Drop rows where price or kilometers are missing

def app():
    st.title("Car Filtering Page")
    st.write("Select filters to refine your car search.")

    df = load_data()

    # Sidebar Filters
    brand = st.sidebar.selectbox("Select Brand", options=['All'] + sorted(df['Brand'].unique().tolist()))
    model = st.sidebar.selectbox("Select Model", options=['All'] + sorted(df['Model'].unique().tolist()))
    variant = st.sidebar.selectbox("Select Variant", options=['All'] + sorted(df['Variant'].unique().tolist()))
    fuel_type = st.sidebar.selectbox("Select Fuel Type", options=['All'] + sorted(df['Fuel Type'].unique().tolist()))
    transmission = st.sidebar.selectbox("Select Transmission", options=['All'] + sorted(df['Transmission'].unique().tolist()))
    owner = st.sidebar.selectbox("Select Owner Type", options=['All'] + sorted(df['Owner'].unique().tolist()))
    body_type = st.sidebar.selectbox("Select Body Type", options=['All'] + sorted(df['Body Type'].unique().tolist()))
    city = st.sidebar.selectbox("Select City", options=['All'] + sorted(df['City'].unique().tolist()))

    # Price Range Filter
    min_price, max_price = int(df['Price'].min()), int(df['Price'].max())
    price_range = st.sidebar.slider("Select Price Range", min_price, max_price, (min_price, max_price))

    # Kilometers Driven Filter
    min_kms, max_kms = int(df['Kilometers'].min()), int(df['Kilometers'].max())
    kms_range = st.sidebar.slider("Select Kilometers Driven", min_kms, max_kms, (min_kms, max_kms))

    # Apply filters
    if brand != 'All':
        df = df[df['Brand'] == brand]
    if model != 'All':
        df = df[df['Model'] == model]
    if variant != 'All':
        df = df[df['Variant'] == variant]
    if fuel_type != 'All':
        df = df[df['Fuel Type'] == fuel_type]
    if transmission != 'All':
        df = df[df['Transmission'] == transmission]
    if owner != 'All':
        df = df[df['Owner'] == owner]
    if body_type != 'All':
        df = df[df['Body Type'] == body_type]
    if city != 'All':
        df = df[df['City'] == city]

    df = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]
    df = df[(df['Kilometers'] >= kms_range[0]) & (df['Kilometers'] <= kms_range[1])]

    # Display filtered results
    st.write("### Filtered Cars")
    st.dataframe(df[['Brand', 'Model', 'Variant', 'Price', 'Fuel Type', 'Transmission', 'Owner', 'Body Type', 'Kilometers', 'City']])

if __name__ == "__main__":
    app()