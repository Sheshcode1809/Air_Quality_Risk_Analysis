import plotly.graph_objects as go
import numpy as np
import streamlit as st
from utils.model_loader import load_model
from utils.predictor import predict_aqi
from utils.live_aqi import get_live_aqi, convert_aqi_scale

def show():
    model = load_model()
    
    st.title("🌫 AQI Dashboard")
    
    city = st.selectbox(
        "Select City",
        ["Ahmedabad", "Delhi", "Mumbai"]
    )
    
    city_coords = {
        "Ahmedabad" : (23.0225,72.5714),
        "Delhi" : (28.7041,77.1025),
        "Mumbai" : (19.0760,72.8777)
    }
    
    lat,lon = city_coords[city]

    st.session_state["lat"] = lat
    st.session_state["lon"] = lon
    st.session_state["city"] = city

    aqi_raw, pollutants = get_live_aqi(lat, lon)
    
    current_aqi = convert_aqi_scale(aqi_raw)
    
    pm25 = pollutants["pm2_5"]
    pm10 = pollutants["pm10"]
    no2 = pollutants["no2"]
    co = pollutants["co"]
    o3 = pollutants["o3"]
    
    highest_pollutant = max(pollutants, key=pollutants.get)

    def classify_aqi(aqi):
    
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    risk_level = classify_aqi(current_aqi)

# ---------------- KPI ROW ----------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current AQI", current_aqi, "+12")
    col2.metric("Highest Pollutant", highest_pollutant)
    col3.metric("Risk Level", risk_level)

    st.divider()

    # ---------------- AQI GAUGE ----------------
    st.subheader("Current AQI Level")
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
    # ---------------- POLLUTANT CARDS ----------------
    st.subheader("Pollutant Breakdown")

    p1, p2, p3, p4, p5 = st.columns(5)

    p1.metric("PM2.5", 120, "+5")
    p2.metric("PM10", 98, "-3")
    p3.metric("NO2", 45, "+2")
    p4.metric("CO", 1.2)
    p5.metric("O3", 30)

    st.divider()

    # ---------------- FORECAST CHART ----------------
    st.subheader("7-Day AQI Forecast")

    forecast = np.random.randint(100, 250, 7)

    st.line_chart(forecast)

    st.divider()

    # ---------------- RISK ALERT ----------------
    if current_aqi > 150:
        st.error("⚠ Air Quality is Unhealthy. Avoid outdoor activities.")
    elif current_aqi > 100:
        st.warning("Moderate air quality. Sensitive groups should be cautious.")
    else:
        st.success("Air quality is Good.")