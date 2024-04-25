import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('mnb_spam_detector.pkl','rb'))

# Define the input features
input_features = ["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina", "oldpeak", "slope"]

# Define the feature labels
feature_labels = {
    "age": "How old are you? ",
    "sex": "What is your gender? ",
    "chest pain type": "What type of chest pain do you experience? (Typical angina, Atypical angina, Non-anginal pain, Asymptomatic) ",
    "resting bp s": "What is your resting blood pressure (in mmHg)?",
    "cholesterol": "What is your cholesterol level (in mg/dL)?",
    "fasting blood sugar": "What is your fasting blood sugar level in mg/dL?",
    "resting ecg": "What is the result of your resting ECG? (Normal, Abnormal with ST-T wave changes)",
    "max heart rate": "What is your maximum heart rate (in beats per minute)?",
    "exercise angina": "Do you experience angina during exercise?",
    "oldpeak": "What is your ST segment depression during exercise relative to rest (oldpeak)?",
    "slope": "What is the slope of the peak exercise ST segment? (Upsloping, Flat, Downsloping) ",
}

input_values = []

# Collect input values
for feature, label in feature_labels.items():
    if feature in ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]:
        # If the feature requires numerical input
        value = st.number_input(f"{label}", step=1)
    else:
        # If the feature requires string input
        value = st.selectbox(f"{label}", [""] + list(feature_labels.values())).lower()
        
    input_values.append(value)


# Convert input values to desired format
converted_values = []

# Convert Sex
sex = 1 if input_values[feature_labels.index("sex")] == "male" else 0
converted_values.append(sex)

# Convert Chest Pain Type
chest_pain_map = {"typical angina": 1, "atypical angina": 2, "non-anginal pain": 3, "asymptomatic": 4}
chest_pain = chest_pain_map.get(input_values[feature_labels.index("chest pain type")])
converted_values.append(chest_pain)

# Convert Fasting Blood Sugar
fasting_blood_sugar = 1 if input_values[feature_labels.index("fasting blood sugar")] > 120 else 0
converted_values.append(fasting_blood_sugar)

# Convert Resting Electrocardiogram Results
ecg_map = {"normal": 0, "having ST-T wave": 1}
resting_ecg = ecg_map.get(input_values[feature_labels.index("resting ecg")])
converted_values.append(resting_ecg)

# Convert Exercise Induced Angina
exercise_angina = input_values[feature_labels.index("exercise angina")]
converted_values.append(exercise_angina)

# Convert Slope of the Peak Exercise ST Segment
slope_map = {"upsloping": 1, "flat": 2, "downsloping": 3}
slope = slope_map.get(input_values[feature_labels.index("slope")])
converted_values.append(slope)
# Perform prediction when the user clicks the button
if st.button('Predict'):
    # Convert input_values to numpy array and reshape
    input_data = np.array(input_values)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction result
    if prediction == 0:
        st.header("No Disease Detected")
    elif prediction == 1:
        st.header("You have a High Risk of Disease")