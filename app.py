import streamlit as st
import pandas as pd
import pickle

# Load model + encoder
with open("churn_rf_healthy_meals.pkl", "rb") as f:
    model = pickle.load(f)
with open("churn_encoder_healthy_meals.pkl", "rb") as f:
    encoder = pickle.load(f)

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

age = st.number_input("Age", min_value=0)
income_level = st.radio("Income Level", ["low", "medium", "high", "very high"])
education = st.radio("Education", ["high school", "other", "graduate", "post graduate"])
device_type = st.radio("Device Type", ["multi-device", "mobile-only", "desktop-only"])
tech_comfort_score = st.number_input("Tech Comfort Score")

if st.button("Predict"):
    raw = pd.DataFrame([{
        "INCOME_LEVEL": income_level,
        "EDUCATION": education,
        "DEVICE_TYPE": device_type,
    }])
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    numeric_df = pd.DataFrame([{
        "AGE": age,
        "TECH_COMFORT_SCORE": tech_comfort_score,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.success(f"Renewal Probability: {probability:.2f} | Churn Risk: {risk}")
