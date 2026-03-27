import joblib
import streamlit as st

@st.cache_resource
def load_model():
    model = joblib.load("aqi_random_forest.pkl")
    return model