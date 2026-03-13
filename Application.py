import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("csat_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("DeepCSAT Score Prediction System")
st.write("Predict Customer Satisfaction Score using Deep Learning")

st.header("Enter Customer Support Details")

# ----------- 19 INPUT FEATURES -----------

f1 = st.number_input("Item Price")
f2 = st.number_input("Connected Handling Time")
f3 = st.number_input("Order Value")
f4 = st.number_input("Order Processing Time")
f5 = st.number_input("Issue Report Time")
f6 = st.number_input("Issue Response Time")
f7 = st.number_input("Survey Response Delay")

# categorical input
agent_shift = st.selectbox("Agent Shift", ["Morning","Afternoon","Night"])
shift_map = {"Morning":0,"Afternoon":1,"Night":2}
f8 = shift_map[agent_shift]

f9 = st.number_input("Agent Tenure")
f10 = st.number_input("Agent Experience Level")
f11 = st.number_input("Supervisor Rating")
f12 = st.number_input("Manager Rating")
f13 = st.number_input("Customer Interaction Count")
f14 = st.number_input("Previous Complaint Count")
f15 = st.number_input("Customer Loyalty Score")
f16 = st.number_input("Product Category Code")
f17 = st.number_input("Customer City Code")
f18 = st.number_input("Channel Code")
f19 = st.number_input("Sub Category Code")

# ----------- Prediction -----------

if st.button("Predict CSAT Score"):

    input_data = np.array([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,
                            f11,f12,f13,f14,f15,f16,f17,f18,f19]])

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"Predicted CSAT Score: {prediction[0][0]:.2f}")