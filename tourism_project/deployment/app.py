import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("Wellness Tourism Package Purchase Prediction")
st.write("Enter the customer and interaction details below to predict whether the customer will purchase the Wellness Tourism Package.")

@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    return joblib.load(model_path)

model = load_trained_model()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=36)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=22500.0, step=500.0)

with col2:
    st.subheader("Travel & Lifestyle")
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    num_children = st.number_input("Number of Children Visiting (Below Age 5)", min_value=0, max_value=10, value=1)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    num_trips = st.number_input("Average Annual Trips", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
    passport = st.selectbox("Holds Valid Passport?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

with col3:
    st.subheader("Sales Pitch Interaction")
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (Minutes)", min_value=1.0, max_value=180.0, value=15.0, step=1.0)
    num_followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=20.0, value=4.0, step=1.0)
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

input_df = pd.DataFrame([{
    "Age": float(age),
    "TypeofContact": typeof_contact,
    "CityTier": int(city_tier),
    "DurationOfPitch": float(duration_of_pitch),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(num_persons),
    "NumberOfFollowups": float(num_followups),
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": float(preferred_star),
    "MaritalStatus": marital_status,
    "NumberOfTrips": float(num_trips),
    "Passport": int(passport),
    "PitchSatisfactionScore": int(pitch_score),
    "OwnCar": int(own_car),
    "NumberOfChildrenVisiting": float(num_children),
    "Designation": designation,
    "MonthlyIncome": float(monthly_income)
}])

st.markdown("---")
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"**Prediction: Yes (Likely to Purchase)** — Purchase Probability: {probability:.2%}")
    else:
        st.warning(f"**Prediction: No (Unlikely to Purchase)** — Purchase Probability: {probability:.2%}")
