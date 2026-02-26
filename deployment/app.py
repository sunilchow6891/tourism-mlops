
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide")
st.title("🚢 Visit with Us: Wellness Package Prediction")

# Configuration
USER_NAME = "sunchow"
MODEL_REPO = f"{USER_NAME}/tourism-model"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.joblib")
    return joblib.load(model_path)

model = load_model()

# Form for all 18 required features
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 18, 100, 30)
    type_contact = st.selectbox("Type of Contact (0: Invited, 1: Self)", [0, 1])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    pitch_dur = st.number_input("Duration of Pitch", 0, 120, 20)
    occupation = st.selectbox("Occupation (0: Salaried, 1: Small Biz, 2: Large Biz, 3: Freelance)", [0, 1, 2, 3])
    gender = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
with col2:
    num_visitors = st.number_input("Persons Visiting", 1, 10, 2)
    pref_star = st.slider("Hotel Star Rating", 3, 5, 3)
    marital = st.selectbox("Marital (0: Single, 1: Married, 2: Divorced)", [0, 1, 2])
    num_trips = st.number_input("Annual Trips", 1, 20, 3)
    passport = st.selectbox("Passport (0: No, 1: Yes)", [0, 1])
    own_car = st.selectbox("Owns Car (0: No, 1: Yes)", [0, 1])
with col3:
    children = st.number_input("Children Visiting", 0, 5, 0)
    designation = st.selectbox("Designation (0: Exec, 1: Mgr, 2: VP, 3: AVP, 4: Dir)", [0, 1, 2, 3, 4])
    income = st.number_input("Monthly Income", 0, 150000, 25000)
    pitch_score = st.slider("Pitch Score", 1, 5, 3)
    prod_pitched = st.selectbox("Product (0: Basic, 1: Delx, 2: Std, 3: S-Delx, 4: King)", [0, 1, 2, 3, 4])
    followups = st.number_input("Follow-ups", 0, 10, 3)

# EXACT COLUMN ORDER FROM DATA DICTIONARY
feature_cols = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus', 
    'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 
    'Designation', 'MonthlyIncome', 'PitchSatisfactionScore', 
    'ProductPitched', 'NumberOfFollowups'
]

input_df = pd.DataFrame([[
    age, type_contact, city_tier, pitch_dur, occupation,
    gender, num_visitors, pref_star, marital,
    num_trips, passport, own_car, children,
    designation, income, pitch_score,
    prod_pitched, followups
]], columns=feature_cols)

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success("Likely to Purchase" if prediction[0] == 1 else "Unlikely to Purchase")
