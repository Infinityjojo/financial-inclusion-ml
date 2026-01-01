import streamlit as st
import pandas as pd
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "bank_account_pipeline.pkl")

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Financial Inclusion Predictor", layout="centered")

st.title("🏦 Financial Inclusion in Africa")
st.write("Predict the likelihood of an individual owning a bank account.")

st.divider()

# --- INPUT FORM ---
with st.form("prediction_form"):
    country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
    year = st.number_input("Survey Year", 2016, 2022, 2018)
    location_type = st.selectbox("Location Type", ["Rural", "Urban"])
    cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
    household_size = st.number_input("Household Size", 1, 20, 4)
    age = st.number_input("Age", 16, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    relationship = st.selectbox(
        "Relationship with Household Head",
        ["Head of Household", "Spouse", "Child", "Other relative", "Non-relative"]
    )
    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced", "Widowed", "Separated"]
    )
    education = st.selectbox(
        "Education Level",
        [
            "No formal education",
            "Primary education",
            "Secondary education",
            "Tertiary education",
            "Other/Dont know/RTA"
        ]
    )
    job_type = st.selectbox(
        "Job Type",
        [
            "Farming and Fishing",
            "Self employed",
            "Formally employed Government",
            "Formally employed Private",
            "Informally employed",
            "No income",
            "Other income"
        ]
    )

    submitted = st.form_submit_button("Predict")

# --- PREDICTION ---
if submitted:
    input_df = pd.DataFrame([{
        "country": country,
        "year": year,
        "location_type": location_type,
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "age_of_respondent": age,
        "gender_of_respondent": gender,
        "relationship_with_head": relationship,
        "marital_status": marital_status,
        "education_level": education,
        "job_type": job_type
    }])

    probability = model.predict_proba(input_df)[0][1]

    threshold = 0.5

    if probability >= threshold:
        st.success(
            f"""✅ **Likely to have a bank account**

**Probability:** {probability:.2%}
"""
        )
    else:
        st.error(
            f"""❌ **Unlikely to have a bank account**

**Probability:** {probability:.2%}
"""
        )

    st.caption(
        "⚠️ Prediction is based on historical survey data and is for analytical purposes only."
    )
