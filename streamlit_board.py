import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
st.title("🌾 Rice Production Predictor Dashboard")

# Dropdown to select rice type
rice_type = st.selectbox("Select Rice Type", ["boro", "aman", "aush"])
file_map = {
    "boro": "boro.csv",
    "aman": "aman.csv",
    "aush": "aush.csv"
}
data = pd.read_csv(file_map[rice_type])

# Drop rows with missing values in target columns
data = data.dropna(subset=[
    "Yied_Hectare_(M.Ton)",
    "Total_Production(M.Ton)"
])

data = data.dropna()

# Features and Targets
features = ["Temperature", "Humidity", "Rainfall", "Soil Moisture", "Evapotranspiration", "Hectares"]

feature_labels = {
    "Temperature": "🌡️ Temperature (°C)",
    "Humidity": "💧 Humidity (%)",
    "Rainfall": "🌧️ Rainfall (mm)",
    "Soil Moisture": "🌱 Soil Moisture (%)",
    "Evapotranspiration": "💨 Evapotranspiration (mm/day)",
    "Hectares": "🌾 Production Area (Hectares)"
}


X = data[features]
y_yield = data["Yied_Hectare_(M.Ton)"]
y_total = data["Total_Production(M.Ton)"]

# Scale and Train models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_yield = RandomForestRegressor(random_state=0).fit(X_scaled, y_yield)
model_total = RandomForestRegressor(random_state=0).fit(X_scaled, y_total)

# Sliders for user input with custom ranges
st.sidebar.header("Provide environmental and agricultural data below")

# 🔁 Define custom slider ranges for each feature
custom_slider_config = {
    "🌡️ Temperature (°C)": {"min": 0.0, "max": 50.0, "default": 30.0},
    "💧 Humidity (%)": {"min": 0.0, "max": 100.0, "default": 70.0},
    "🌧️ Rainfall (mm)": {"min": 0.0, "max": 1500.0, "default": 400.0},
    "🌱 Soil Moisture (%)": {"min": 0.0, "max": 1.0, "default": 0.5},
    "💨 Evapotranspiration (mm/day)": {"min": 0.0, "max": 20.0, "default": 5.0},
    "🌾 Production Area (Hectares)": {"min": 0.0, "max": 100000.0, "default": 1000.0}  # Include this only if needed
}

def user_inputs():
    inputs = []
    for feature in features:
        label = feature_labels[feature]  # Get the pretty label with emoji
        config = custom_slider_config[label]  # Use pretty label to access config
        value = st.sidebar.slider(
            label=label,
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"]
        )
        inputs.append(value)
    return inputs

# Get input and scale
input_data = np.array(user_inputs()).reshape(1, -1)
input_scaled = scaler.transform(input_data)


# Prediction
predicted_yield = model_yield.predict(input_scaled)[0]
predicted_total = model_total.predict(input_scaled)[0]

# Descriptive lines before showing predictions
st.markdown("🧠 **This model calculates the Yield Per Hectare and Total Production of rice using Machine Learning models.**")
st.markdown("🔽 **According to your provided parameters, your:**")

# Display predictions
st.subheader(f"🌱 Predicted Yield: {predicted_yield:.2f} MT/hectare")
st.subheader(f"📦 Predicted Total Production: {predicted_total:.2f} MT")



# Feature Importance
st.subheader("🔍 Feature Importances")

importances = model_total.feature_importances_

# Build full importance DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

# ✅ Exclude "Hectares" from the plot
importance_df = importance_df[importance_df["Feature"] != "Hectares"]

# Sort and plot
importance_df = importance_df.sort_values(by="Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))
