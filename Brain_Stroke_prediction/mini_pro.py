import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('Sample.pkl', 'rb'))

def preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # Preprocess categorical features
    gender_encoded = 1 if gender == 'Male' else 0  # Assuming male is encoded as 1, female as 0
    ever_married_encoded = 1 if ever_married == 'Yes' else 0  # Assuming yes is encoded as 1, no as 0
    work_type_encoded = 1 if work_type == 'Private' else 0  # Assuming private is encoded as 1, other as 0
    Residence_type_encoded = 1 if Residence_type == 'Urban' else 0  # Assuming urban is encoded as 1, rural as 0
    smoking_status_encoded = 1 if smoking_status == 'Smokes' else 0  # Assuming smokes is encoded as 1, other as 0
    
    # Convert input to numpy array
    input_data = np.array([[gender_encoded, float(age), float(hypertension), float(heart_disease), ever_married_encoded, work_type_encoded, Residence_type_encoded, float(avg_glucose_level), float(bmi), smoking_status_encoded]]).astype(np.float64)
    return input_data

def predict_brain_stroke(input_data):
    prediction = model.predict_proba(input_data)
    pred = '{0:,.2f}'.format(prediction[0][1])
    return float(pred)

def main():
    st.title("Brain Stroke Prediction App")
    html_temp = '''
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">Brain Stroke Prediction App</h2>
    </div>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    # Dropdowns for categorical features
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age")
    hypertension = st.number_input("Hypertension")
    heart_disease = st.number_input("Heart Disease")
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job","Other"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")
    smoking_status = st.selectbox("Smoking Status", ["Smokes","formerly smoked","never smoked","Other"])

    if st.button("Predict"):
        input_data = preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
        output = predict_brain_stroke(input_data)
        st.success("The probability of stroke is {}".format(output))
        if output > 0.5:
            st.error("You may be at risk of stroke!")
        else:
            st.success("You are not at risk of stroke!")

if __name__ == '__main__':
    main()
