import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained K-Means model and the preprocessor
kmeans = joblib.load('kmeans_cluster_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

def user_input_features():
    # Define possible options for each feature based on the dataset
    medication_options = ['Paracetamol', 'Lipitor', 'Ibuprofen', 'Penicillin']
    admission_type_options = ['Urgent', 'Elective', 'Emergency']
    gender_options = ['Male', 'Female']
    medical_condition_options = ['Diabetes', 'Arthritis', 'Asthma', 'Heart Disease', 'Cancer', 'Obesity']
    test_results_options = ['Normal', 'Abnormal', 'Inconclusive']
    blood_type_options = ['A+', 'B+', 'O+', 'AB+', 'A-', 'B-', 'O-', 'AB-']

    age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1, format="%d", placeholder="Enter age")
    billing_amount = st.number_input("Billing Amount", min_value=0.0, value=0.0, step=100.0, format="%.2f", placeholder="Enter billing amount")
    length_of_stay = st.number_input("Length of Stay", min_value=0, max_value=365, value=0, step=1, format="%d", placeholder="Enter length of stay in days")
    medication = st.selectbox('Medication', options=[""] + medication_options, index=0, format_func=lambda x: 'Select medication' if x == "" else x)
    admission_type = st.selectbox('Admission Type', options=[""] + admission_type_options, index=0, format_func=lambda x: 'Select admission type' if x == "" else x)
    gender = st.selectbox('Gender', options=[""] + gender_options, index=0, format_func=lambda x: 'Select gender' if x == "" else x)
    medical_condition = st.selectbox('Medical Condition', options=[""] + medical_condition_options, index=0, format_func=lambda x: 'Select medical condition' if x == "" else x)
    test_results = st.selectbox('Test Results', options=[""] + test_results_options, index=0, format_func=lambda x: 'Select test result' if x == "" else x)
    blood_type = st.selectbox('Blood Type', options=[""] + blood_type_options, index=0, format_func=lambda x: 'Select blood type' if x == "" else x)
    return [age, billing_amount, length_of_stay, medication, admission_type, gender, medical_condition, test_results, blood_type]

st.title("Healthcare Patient Clustering")

# User input features
inputs = user_input_features()

# Ensure all inputs are provided
if all(inputs):
    if st.button('Predict Cluster'):
        input_df = pd.DataFrame([inputs], columns=['Age', 'Billing Amount', 'Length of Stay', 'Medication', 'Admission Type', 'Gender', 'Medical Condition', 'Test Results', 'Blood Type'])
        input_transformed = preprocessor.transform(input_df)
        prediction = kmeans.predict(input_transformed)
        predicted_cluster = prediction[0]
        
        st.write(f'The predicted cluster is: {predicted_cluster}')
        
        # Display insights based on the predicted cluster
        if predicted_cluster == 0:
            st.subheader("Cluster 0 Insights")
            st.write("""
            - **Average Age:** Approximately 49.5 years
            - **Average Billing Amount:** $12,946.82
            - **Average Length of Stay:** About 7 days
            - **Common Characteristics:**
              - Medications: Paracetamol
              - Admission Type: Urgent
              - Gender: Predominantly Male
              - Test Results: Abnormal
              - Blood Type: B-
            - **Interpretation:** This cluster typically includes middle-aged males with urgent medical needs, reflected in moderate billing and shorter stays.
            """)
        
        elif predicted_cluster == 1:
            st.subheader("Cluster 1 Insights")
            st.write("""
            - **Average Age:** Approximately 70.2 years
            - **Average Billing Amount:** $36,733.06
            - **Average Length of Stay:** About 14 days
            - **Common Characteristics:**
              - Medications: Lipitor
              - Admission Type: Elective
              - Gender: Predominantly Male
              - Test Results: Abnormal
              - Blood Type: A+
            - **Interpretation:** This cluster often comprises older males undergoing elective procedures, resulting in higher billing and moderate length of stay.
            """)
        
        elif predicted_cluster == 2:
            st.subheader("Cluster 2 Insights")
            st.write("""
            - **Average Age:** Approximately 31.5 years
            - **Average Billing Amount:** $37,676.63
            - **Average Length of Stay:** About 17 days
            - **Common Characteristics:**
              - Medications: Ibuprofen
              - Admission Type: Elective
              - Gender: Predominantly Male
              - Test Results: Normal
              - Blood Type: O+
            - **Interpretation:** Younger males with elective treatments are predominant in this cluster, characterized by higher billing and longer stays.
            """)

        elif predicted_cluster == 3:
            st.subheader("Cluster 3 Insights")
            st.write("""
            - **Average Age:** Approximately 52.7 years
            - **Average Billing Amount:** $13,768.49
            - **Average Length of Stay:** About 23 days
            - **Common Characteristics:**
              - Medications: Penicillin
              - Admission Type: Emergency
              - Gender: Predominantly Female
              - Test Results: Inconclusive
              - Blood Type: A-
            - **Interpretation:** This cluster includes predominantly female patients requiring emergency care, with lower billing but extended stays due to complexity.
            """)
else:
    st.warning("Please fill in all fields before predicting the cluster.")
