# app.py
import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# App UI
st.title("📚 Study Hours vs Test Score Predictor")
st.write("A simple Linear Regression model that predicts your test score based on how many hours you studied.")

# User input
hours = st.number_input("Enter hours studied", min_value=0.0, step=0.5)

# Predict
if st.button("Predict"):
    input_data = np.array([[hours]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Test Score: {prediction[0]:.2f}")
