import pandas as pd
import os
import re  # For text pattern matching

# Define dataset folder path
dataset_folder = r"D:\vinoth\capstone\Car Dekho"

# List of Excel files in the dataset
files = [
    "bangalore_cars.xlsx", "chennai_cars.xlsx", "delhi_cars.xlsx",
    "hyderabad_cars.xlsx", "jaipur_cars.xlsx", "kolkata_cars.xlsx"
]

# Initialize an empty list to store DataFrames
df_list = []

# Function to extract year
def extract_year(text):
    match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', str(text))  # Match years from 1950-2029
    return int(match.group()) if match else None

# Function to extract Kms Driven
def extract_kms(text):
    match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*kms?', str(text), re.IGNORECASE)
    return int(match.group(1).replace(",", "")) if match else None

# Function to extract Fuel Type
def extract_fuel_type(text):
    fuels = ["Petrol", "Diesel", "CNG", "Electric", "LPG", "Hybrid"]
    for fuel in fuels:
        if fuel.lower() in str(text).lower():
            return fuel
    return "Unknown"

# Load all datasets
for file in files:
    file_path = os.path.join(dataset_folder, file)
    df = pd.read_excel(file_path)  # Read Excel file
    df["City"] = file.split("_")[0].capitalize()  # Add a city column based on file name

    # Extract features from 'new_car_detail' or 'new_car_specs'
    df["Year"] = df["new_car_detail"].apply(extract_year)
    df["Kms_Driven"] = df["new_car_specs"].apply(extract_kms)
    df["Fuel_Type"] = df["new_car_specs"].apply(extract_fuel_type)

    df_list.append(df)

# Combine all data into one DataFrame
car_data = pd.concat(df_list, ignore_index=True)

# Fill missing values
car_data["Year"].fillna(0, inplace=True)
car_data["Kms_Driven"].fillna(0, inplace=True)
car_data["Fuel_Type"].fillna("Unknown", inplace=True)

# Save cleaned data to CSV
cleaned_file_path = os.path.join(dataset_folder, "Cleaned_Car_Dekho.csv")
car_data.to_csv(cleaned_file_path, index=False)

print(f"✅ Data cleaning completed. Cleaned file saved at: {cleaned_file_path}")