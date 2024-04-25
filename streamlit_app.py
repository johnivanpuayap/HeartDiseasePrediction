import streamlit as st
from keras.models import load_model
import numpy as np


# Load the trained model
# model = pickle.load(open('heart_disease_detector.pkl','rb'))

model = load_model('heart_disease_predictor.keras')

# Define the input features
input_features = ["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina", "oldpeak", "slope"]

# Define the feature labels
feature_labels = {
    "age": "How old are you? ",
    "sex": "What is your gender? ",
    "chest pain type": "What type of chest pain do you experience?",
    "resting bp s": "What is your resting blood pressure (in mmHg)?",
    "cholesterol": "What is your cholesterol level (in mg/dL)?",
    "fasting blood sugar": "What is your fasting blood sugar level in mg/dL?",
    "resting ecg": "What is the result of your resting ECG?",
    "max heart rate": "What is your maximum heart rate (in beats per minute)?",
    "exercise angina": "Do you experience angina during exercise?",
    "oldpeak": "What is your ST segment depression during exercise relative to rest (oldpeak)?",
    "slope": "What is the slope of the peak exercise ST segment?",
}


st.title('Heart Disease Predictor')

st.write("""
Welcome to the Heart Disease Predictor! This tool uses machine learning to assess the risk of heart disease based on your symptoms. Simply fill out the form below and click "Predict" to get your result.
""")

input_values = []

# Collect input values
for feature, label in feature_labels.items():
    if feature in ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak", "fasting blood sugar"]:
        # If the feature requires numerical input
        value = st.number_input(f"{label}", step=1)
    elif feature == "sex":
        # If the feature is gender
        value = st.selectbox(f"{label}", ["Male", "Female"])
    elif feature == "chest pain type":
        # If the feature is chest pain type
        value = st.selectbox(f"{label}", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    elif feature == "resting ecg":
        # If the feature is resting ECG result
        value = st.selectbox(f"{label}", ["Normal", "Abnormal with ST-T wave changes", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
    elif feature == "exercise angina":
        # If the feature is exercise-induced angina
        value = st.selectbox(f"{label}", ["Yes", "No"])
    elif feature == "slope":
        # If the feature is the slope of the peak exercise ST segment
        value = st.selectbox(f"{label}", ["Upsloping", "Flat", "Downsloping"])
    else:
        # Default case
        value = st.text_input(f"{label}")
        
    input_values.append(value)



# Convert Sex
input_values[input_features.index("sex")] = 1 if input_values[input_features.index("sex")] == "Male" else 0

# Convert Chest Pain Type
chest_pain_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
input_values[input_features.index("chest pain type")] = chest_pain_map.get(input_values[input_features.index("chest pain type")])

# Convert Fasting Blood Sugar
input_values[input_features.index("fasting blood sugar")] = 1 if input_values[input_features.index("fasting blood sugar")] > 120 else 0

# Convert Resting Electrocardiogram Results
ecg_map = {"Normal": 0, "Abnormal with ST-T wave changes": 1, "Showing probable or definite left ventricular hypertrophy by Estes' criteria": 2}
input_values[input_features.index("resting ecg")] = ecg_map.get(input_values[input_features.index("resting ecg")])

# Convert Exercise Induced Angina
input_values[input_features.index("exercise angina")] = 1 if input_values[input_features.index("exercise angina")] == "Yes" else 0

# Convert Slope of the Peak Exercise ST Segment
slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
input_values[input_features.index("slope")] = slope_map.get(input_values[input_features.index("slope")])


# Perform prediction when the user clicks the button
if st.button('Predict Heart Disease'):
    # Convert input_values to a numpy array and reshape
    input_data = np.array(input_values).reshape(1,-1)
    
    # Make prediction
    prediction = model.predict(input_data)

    print(prediction)

    result = int(round(prediction[0][0]))
    
    # Display prediction result
    if result == 0:
        st.success("No signs of heart disease were detected based on the provided information.")
    elif result == 1:
        st.error("Heart disease was detected based on the provided information.")