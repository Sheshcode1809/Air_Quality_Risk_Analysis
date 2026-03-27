import streamlit as st
from streamlit_option_menu import option_menu
import Dashboard
import Forecast
import Risk_Analysis

st.set_page_config(
    page_title="AQI Intelligence System",
    layout="wide"
)

st.title("🌍 Air Quality Prediction & Risk Analysis")

st.write("""
This system predicts AQI using Machine Learning models
and provides pollution risk analysis.
""")

with st.sidebar:
    selected = option_menu(
        menu_title = "AQI Intelligence",
        options=["Dashboard", "Forecast", "Risk Analysis"],
        icons=["speedometer", "graph-up", "exclamation-triangle", "gear"],
        default_index=0,
    )

if selected == "Dashboard":
    Dashboard.show()

elif selected == "Forecast":
    Forecast.show()

elif selected == "Risk Analysis":
    Risk_Analysis.show()
