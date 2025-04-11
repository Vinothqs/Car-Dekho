import joblib

# Load your trained model
model = joblib.load("car_price_model.pkl")

# Print the feature names it was trained on
print(model.feature_names_in_)