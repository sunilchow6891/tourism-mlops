
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Predictor", layout="centered")

st.title("🚢 Visit with Us: Package Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Package.")

# Configuration (Update these!)
USER_NAME = "sunchow"
MODEL_REPO = f"{USER_NAME}/tourism-model"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.joblib")
    return joblib.load(model_path)

model = load_model()

# Create Input Fields based on Data Dictionary
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    trips = st.number_input("Number of Trips annually", 1, 20, 3)

with col2:
    income = st.number_input("Monthly Income", 0, 100000, 25000)
    passport = st.selectbox("Passport (0=No, 1=Yes)", [0, 1])
    property_star = st.slider("Preferred Property Star", 3, 5, 3)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    followups = st.number_input("Number of Follow-ups", 0, 10, 3)

# Additional fields to match the training feature count (18 columns used in Step 3)
# Note: In a production app, you would include all inputs from the Data Dictionary
input_data = pd.DataFrame([[age, city_tier, trips, passport, property_star, pitch_score, followups, income]], 
                          columns=['Age', 'CityTier', 'NumberOfTrips', 'Passport', 
                                   'PreferredPropertyStar', 'PitchSatisfactionScore', 
                                   'NumberOfFollowups', 'MonthlyIncome'])

# Dummy encoding/alignment for the sake of example consistency
# In your real code, ensure categorical columns are encoded exactly as in Step 3

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎯 Result: High Probability of Purchase!")
    else:
        st.error("📉 Result: Low Probability of Purchase.")
