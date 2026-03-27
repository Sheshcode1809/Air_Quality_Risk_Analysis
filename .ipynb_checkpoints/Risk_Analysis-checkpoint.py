import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
from utils.model_loader import load_model
from utils.predictor import predict_aqi

API_KEY = "395c63a31b5fc377f2d9458689350218"

def show():
    # df = pd.read_csv("air_quality_with_risk.csv")
    # data = st.dataframe(df)
    
    st.title("⚠ AQI Risk Analysis")
    st.set_page_config(layout="wide")
    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    city = st.session_state["city"]

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)

    data = response.json()   # JSON data stored here

    pollution = data["list"][0]["components"]
    st.write(pollution)
    
    pm25 = pollution["pm2_5"]
    pm10 = pollution["pm10"]
    no2 = pollution["no2"]
    
    st.metric("PM2.5", pm25)
    st.metric("PM10", pm10)
    st.metric("NO2", no2)
    st.markdown("Analyze health risk levels based on current and predicted AQI.")
    
    # ---------------- SAMPLE DATA (Replace Later With Real Data) ----------------
    current_aqi = 185
    
    pollutants = {
        "PM2.5": 130,
        "PM10": 95,
        "NO2": 50,
        "CO": 1.4,
        "O3": 35
    }
    
    # ---------------- RISK CLASSIFICATION FUNCTION ----------------
    def classify_risk(aqi):
        if aqi <= 50:
            return "Good", "Low"
        elif aqi <= 100:
            return "Moderate", "Medium"
        elif aqi <= 200:
            return "Unhealthy", "High"
        else:
            return "Hazardous", "Critical"
    
    category, level = classify_risk(current_aqi)
    
    # ---------------- RISK SUMMARY ----------------
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.metric("Current AQI", current_aqi)
        st.metric("Risk Category", category)
        st.metric("Risk Level", level)
    
    with col2:
        st.subheader("Risk Meter")
    
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            gauge={
                "axis": {"range": [0, 500]},
                "steps": [
                    {"range": [0, 50], "color": "green"},
                    {"range": [50, 100], "color": "yellow"},
                    {"range": [100, 200], "color": "orange"},
                    {"range": [200, 300], "color": "red"},
                ],
            }
        ))
    
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ---------------- POLLUTANT CONTRIBUTION ----------------
    st.subheader("Pollutant Contribution Analysis")
    
    df_pollutants = pd.DataFrame({
        "Pollutant": list(pollutants.keys()),
        "Value": list(pollutants.values())
    })
    
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        x=df_pollutants["Pollutant"],
        y=df_pollutants["Value"],
        name="Concentration"
    ))
    
    fig_bar.update_layout(
        xaxis_title="Pollutant",
        yaxis_title="Concentration Level",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()
    
    # ---------------- SENSITIVE GROUP ALERTS ----------------
    st.subheader("Sensitive Group Advisory")
    
    if current_aqi > 150:
        st.error("⚠ Children and elderly should avoid outdoor exposure.")
        st.error("⚠ Asthma patients must wear protective masks.")
    elif current_aqi > 100:
        st.warning("Sensitive groups should limit outdoor activity.")
    else:
        st.success("Air quality acceptable for most individuals.")
    
    st.divider()
    
    # ---------------- PREVENTIVE RECOMMENDATIONS ----------------
    st.subheader("Recommended Precautions")
    
    recommendations = []
    
    if current_aqi > 200:
        recommendations = [
            "Avoid outdoor activities.",
            "Wear N95 masks.",
            "Use air purifiers indoors.",
            "Keep windows closed."
        ]
    elif current_aqi > 150:
        recommendations = [
            "Limit outdoor exercise.",
            "Wear masks if going outside.",
            "Monitor AQI regularly."
        ]
    elif current_aqi > 100:
        recommendations = [
            "Sensitive individuals should reduce prolonged exposure."
        ]
    else:
        recommendations = [
            "Air quality is safe. Normal activities allowed."
        ]
    
    for rec in recommendations:
        st.write("•", rec)
    
    st.subheader("Health Advisory")
    
    if avg > 200:
        st.error("Avoid outdoor activities. Air pollution extremely high.")
    elif avg > 150:
        st.warning("Sensitive groups should limit outdoor exposure.")
    else:
        st.success("Air quality expected to remain moderate.")

