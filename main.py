import streamlit as st
import joblib
import numpy as np

# Define default values for the missing features
default_values = {
    "Father's Education": 1,
    "Mother's Education": 1,
    "Parental Income": 1,
    "Number of immediate family members": 5,
    "Any close family member with the same profession available for guidance": 1,
    "Availing any scholarship": 0,
    "Basic Education Stream": 1,
    "Intermediate Stream": 1,
    "Matric percentage": 75,
    "ID No.": 1,
    "I opted for this program of study because of my own interest": 1
}

# List of available models
model_files = ['DecisionTreeModel.joblib', 'K-Neigbhors.joblib', 'Lasso.joblib', 'Linear_Regression.joblib', 'RandomForestRegression.joblib']

def predict_performance(intermediate_percentage, sgpa_1st, sgpa_2nd, sgpa_3rd, sgpa_4th, selected_model_file):
    try:
        # Load the selected model
        model = joblib.load(f'{selected_model_file}')

        # Prepare data for prediction with default values
        input_data = np.array([
            [
                default_values["Father's Education"],
                default_values["Mother's Education"],
                default_values["Parental Income"],
                default_values["Number of immediate family members"],
                default_values["Any close family member with the same profession available for guidance"],
                default_values["Availing any scholarship"],
                default_values["Basic Education Stream"],
                default_values["Intermediate Stream"],
                default_values["Matric percentage"],
                intermediate_percentage,
                sgpa_1st, sgpa_2nd, sgpa_3rd, sgpa_4th,
                default_values["ID No."],
                default_values["I opted for this program of study because of my own interest"]
            ]
        ])

        # Make prediction
        prediction = model.predict(input_data)[0]
        sgpa_5th, cgpa_5th = prediction

        # Determine the performance comment for SGPA
        sgpa_comment = get_performance_comment(sgpa_5th)

        # Determine the performance comment for CGPA
        cgpa_comment = get_performance_comment(cgpa_5th)

        # Display results
        st.success(f"Predicted SGPA for 5th Semester: {sgpa_5th:.2f} ({sgpa_comment})")
        st.success(f"Predicted CGPA for 5th Semester: {cgpa_5th:.2f} ({cgpa_comment})")
    except ValueError as e:
        st.error(f"Input Error: {str(e)}")

def get_performance_comment(gpa):
    if 3.51 <= gpa <= 4.00:
        return "Extraordinary Performance"
    elif 3.00 <= gpa < 3.51:
        return "Very Good Performance"
    elif 2.51 <= gpa < 3.00:
        return "Good Performance"
    elif 2.00 <= gpa < 2.51:
        return "Satisfactory Performance"
    elif 1.00 <= gpa < 2.00:
        return "Poor Performance"
    elif 0.00 <= gpa < 1.00:
        return "Very Poor Performance"
    else:
        return "Invalid GPA value"

# Streamlit UI
st.title("GPA Predictor")

# User inputs
intermediate_percentage = st.number_input("Intermediate Percentage", min_value=0.0, max_value=100.0, value=75.0)
sgpa_1st = st.number_input("SGPA in BS First Semester", min_value=0.0, value=0.0)
sgpa_2nd = st.number_input("SGPA in BS Second Semester", min_value=0.0, value=0.0)
sgpa_3rd = st.number_input("SGPA in BS Third Semester", min_value=0.0, value=0.0)
sgpa_4th = st.number_input("SGPA in BS Fourth Semester", min_value=0.0, value=0.0)
selected_model_file = st.selectbox("Select Model", model_files)

if st.button("Predict"):
    predict_performance(intermediate_percentage, sgpa_1st, sgpa_2nd, sgpa_3rd, sgpa_4th, selected_model_file)
