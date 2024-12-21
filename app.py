import streamlit as st
import pickle
import numpy as np

# Load the models
with open('stacked_model_cgpa.pkl', 'rb') as file:
    model_cgpa = pickle.load(file)

with open('stacked_model_sgpa.pkl', 'rb') as file:
    model_sgpa = pickle.load(file)

# Streamlit interface
st.title("CGPA and SGPA Prediction System")

# Sidebar for additional information
with st.sidebar:
    st.header("About")
    st.info("This application predicts the SGPA and CGPA for BS Fifth Semester based on academic history.")
    st.header("Instructions")
    st.write("1. Fill in the academic details in the main section.")
    st.write("2. Provide additional information in the checkboxes and dropdowns.")
    st.write("3. Click on 'Predict' to see the predictions.")

# Main layout
st.markdown("## Enter Academic Details")
col1, col2 = st.columns(2)

with col1:
    matric_percentage = st.number_input("Matric Percentage", min_value=0.0, max_value=100.0, format="%.2f")
    sgpa_1st_sem = st.number_input("SGPA in BS First Semester", min_value=0.0, max_value=4.0, format="%.2f")
    sgpa_3rd_sem = st.number_input("SGPA in BS Third Semester", min_value=0.0, max_value=4.0, format="%.2f")

with col2:
    intermediate_percentage = st.number_input("Intermediate Percentage", min_value=0.0, max_value=100.0, format="%.2f")
    sgpa_2nd_sem = st.number_input("SGPA in BS Second Semester", min_value=0.0, max_value=4.0, format="%.2f")
    sgpa_4th_sem = st.number_input("SGPA in BS Fourth Semester", min_value=0.0, max_value=4.0, format="%.2f")

st.markdown("## Additional Information")
gender = st.radio("Gender", ["Male", "Female"])
place_of_birth = st.selectbox("Place of Birth", ["Balochistan", "Punjab", "KPK", "Sindh", "Federal Capital", "International"])
fathers_education = st.selectbox("Father's Education", ["BS", "Intermediate", "MS", "PhD"])
mothers_education = st.selectbox("Mother's Education", ["BS", "Intermediate", "MS", "PhD"])
parental_income = st.slider("Parental Income", 50000, 300000, step=10000)
family_members = st.slider("Number of Immediate Family Members", 3, 9)
family_member_profession = st.checkbox("Any Close Family Member with the Same Profession for Guidance")
scholarship = st.checkbox("Availing Any Scholarship")
program_interest = st.selectbox("I Opted for This Program of Study Because of My Own Interest", ["Agree", "Strong Agree", "Neutral", "Disagree", "Strong Disagree"])

# Prediction
if st.button("Predict SGPA and CGPA"):
    input_data = np.array([matric_percentage, intermediate_percentage, sgpa_1st_sem, sgpa_2nd_sem, sgpa_3rd_sem, sgpa_4th_sem]).reshape(1, -1)

    sgpa_prediction = model_sgpa.predict(input_data)
    cgpa_prediction = model_cgpa.predict(input_data)

    st.markdown("## Prediction Results")
    st.success(f"Predicted SGPA in BS Fifth Semester: {sgpa_prediction[0]:.2f}")
    st.success(f"Predicted CGPA in BS Fifth Semester: {cgpa_prediction[0]:.2f}")

# Run the app: streamlit run your_script_name.py
