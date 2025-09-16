import streamlit as st
import pandas as pd
import joblib

# Load model and column order
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Exam Score Predictor")

# --- Collect raw user inputs ---
gender = st.selectbox("Gender", ["Male","Female"])
part_time_job = st.selectbox("Part-time job?", ["Yes","No"])
absence_days = st.number_input("Absence days", min_value=0)
extracurricular = st.selectbox("Extracurricular activities?", ["Yes","No"])
weekly_hours = st.number_input("Weekly self-study hours", min_value=0.0)
career = st.selectbox("Career aspiration", ["Science","Commerce","Arts"])

# --- Put them in a dict using the *same* feature names as in training ---
input_dict = {
    'gender': gender,
    'part_time_job': part_time_job,
    'absence_days': absence_days,
    'extracurricular_activities': extracurricular,
    'weekly_self_study_hours': weekly_hours,
    'career_aspiration': career
}

# Create a 1-row DataFrame
input_df = pd.DataFrame([input_dict])

# --- Apply the SAME one-hot encoding ---
input_encoded = pd.get_dummies(input_df, drop_first=True)

# --- Reindex so columns match training columns exactly ---
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# --- Predict ---
if st.button("Predict Exam Score"):
    pred = model.predict(input_encoded)[0]
    st.success(f"Predicted Exam Score: {pred:.2f}")


