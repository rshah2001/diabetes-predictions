import streamlit as st

st.set_page_config(page_title="Diabetes Prediction Studio", layout="wide")
st.title("Diabetes Prediction Studio")
st.write(
    "Explore features → compare ML models → evaluate decision impact → predict risk → see it in geographic context."
)

st.info(
    "The Pima diabetes dataset and a pre-trained model are built in — every page works "
    "immediately. Use the left sidebar: Insights → Model Compare → Decision Impact → "
    "Predict → Geo Context. (Upload is only needed for your own CSV.)"
)
