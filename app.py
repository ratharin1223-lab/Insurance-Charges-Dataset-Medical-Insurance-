import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Ensure you point to the correct path where src/train.py saved the model
# Example: model = joblib.load('models/insurance_model.pkl') 
# For this script to work immediately, we will simulate the loading or you must update the path
try:
    model = joblib.load('insurance_model.pkl') 
except:
    st.error("Model file not found. Please run src/train.py to generate 'insurance_model.pkl' first.")
    st.stop()

def main():
    # Set page configuration
    st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥")

    # App Title and Description
    st.title("🏥 Medical Insurance Charge Predictor")
    st.write("Enter patient details below to get an estimated insurance premium.")

    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            children = st.slider("Number of Children", 0, 10, 0)
        
        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            smoker = st.selectbox("Smoker", ["yes", "no"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
        
        submit_btn = st.form_submit_button("Predict Charges")

    # Processing and Prediction
    if submit_btn:
        # Create a dataframe for the model
        # Note: You must ensure these inputs match exactly how you trained the model in notebooks/analysis.ipynb
        # If you used LabelEncoding or OneHotEncoding, you need to apply that here before predicting.
        
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        # PREPROCESSING PLACEHOLDER
        # If your model expects numbers (0/1) instead of strings ("yes"/"no"), map them here:
        # input_data['smoker'] = input_data['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
        # input_data['sex'] = input_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        # Handle OneHotEncoding for region if necessary
        
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Insurance Charge: **${prediction:,.2f}**")
            
            # Optional: Visual Context
            st.info("Note: Smoking and high BMI are significant drivers of higher costs.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()
