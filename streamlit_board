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

# Features and Targets
features = [
    "Temperature", "Humidity", "Rainfall",
    "Soil Moisture",
    "Evapotranspiration", "Hectares"
]
X = data[features]
y_yield = data["Yied_Hectare_(M.Ton)"]
y_total = data["Total_Production(M.Ton)"]

# Scale and Train models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_yield = RandomForestRegressor(random_state=0).fit(X_scaled, y_yield)
model_total = RandomForestRegressor(random_state=0).fit(X_scaled, y_total)

# Sliders for user input
st.sidebar.header("Input Parameters")
def user_inputs():
    return [st.sidebar.slider(f, float(X[f].min()), float(X[f].max()), float(X[f].mean())) for f in features]
input_data = np.array(user_inputs()).reshape(1, -1)
input_scaled = scaler.transform(input_data)

# Prediction
predicted_yield = model_yield.predict(input_scaled)[0]
predicted_total = model_total.predict(input_scaled)[0]

# Display predictions
st.subheader(f"🌱 Predicted Yield: {predicted_yield:.2f} MT/hectare")
st.subheader(f"📦 Predicted Total Production: {predicted_total:.2f} MT")

# Bar chart of actual vs predicted total production
st.subheader("📊 Actual vs Predicted Total Production")
fig, ax = plt.subplots()
ax.bar(["Predicted", "Mean of Dataset"], [predicted_total, y_total.mean()], color=["orange", "green"])
st.pyplot(fig)

# Feature Importance
st.subheader("🔍 Feature Importances")
importances = model_total.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))
