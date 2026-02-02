import streamlit as st

st.set_page_config(page_title="Diabetes Prediction Studio", layout="wide")
st.title("Diabetes Prediction Studio")
st.write(
    "Upload a diabetes dataset → explore features → compare ML models → evaluate decision impact → predict risk."
)

st.info("Use the left sidebar to navigate: Upload → Insights → Model Compare → Decision Impact → Predict.")
