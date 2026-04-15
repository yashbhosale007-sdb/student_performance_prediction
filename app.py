import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD THE MODEL ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- HEADER ---
st.title("🎓 Student Performance Prediction")
st.write("Input the student metrics below to predict the final outcome.")

# --- INPUT SECTION ---
st.subheader("Student Metrics")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
    study_hours = st.number_input("Study Hours Per Week", 0, 100, 10)
    prev_grade = st.number_input("Previous Grade", 0, 100, 75)

with col2:
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    parental_support = st.selectbox("Parental Support Level", ["Low", "Medium", "High"])
    # Mapping inputs to numerical values as expected by the model
    gender_val = 1 if gender == "Male" else 0
    extra_val = 1 if extracurricular == "Yes" else 0
    support_map = {"Low": 0, "Medium": 1, "High": 2}
    support_val = support_map[parental_support]

# Create the feature array matching the model's training format
# Based on your model metadata: Gender, AttendanceRate, StudyHoursPerWeek, 
# PreviousGrade, ExtracurricularActivities, ParentalSupport, etc.
features = np.array([[gender_val, attendance, study_hours, prev_grade, extra_val, support_val]])

# --- PREDICTION LOGIC ---
st.markdown("---")
if st.button("Predict Performance"):
    prediction = model.predict(features)
    
    st.balloons()
    st.subheader("Result:")
    
    # Check if prediction is a class or a number
    result_text = f"Predicted Grade/Status: {prediction[0]}"
    
    st.markdown(f"""
        <div class="result-box" style="background-color: #e8f5e9; border: 2px solid #4CAF50; color: #2e7d32;">
            {result_text}
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><p style='text-align: center; color: grey;'>Powered by Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)
