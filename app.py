import streamlit as st
import joblib

# load the saved model
model = joblib.load("model.pkl")

st.title("Exam Score Predictor")

hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, step=0.5)
assignments = st.number_input("Assignments completed", min_value=0, step=1)

if st.button("Predict Score"):
    pred = model.predict([[hours, assignments]])[0]
    st.success(f"Predicted score: {pred:.2f}")
