import streamlit as st
import pandas as pd
import requests
from utils.model_loader import load_model
from utils.predictor import predict_aqi

API_KEY = "395c63a31b5fc377f2d9458689350218"

def show():
    st.title("📈 AQI Forecasting")
    
    city = st.session_state.get("city", "Unknown")

    st.subheader(f"AQI Forecast for {city}")
    
    if "lat" not in st.session_state:
        st.warning("Please select a city in Dashboard first.")
        return

    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    city = st.session_state["city"]

    st.subheader(f"7-Day AQI Forecast for {city}")

    # Load ML model
    model = load_model()

    # Fetch forecast data
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"

    response = requests.get(url)
    data = response.json()

    forecast_list = data["list"][:7]

    predictions = []
    dates = []

    for item in forecast_list:

        comp = item["components"]

        pm25 = comp["pm2_5"]
        pm10 = comp["pm10"]
        no2 = comp["no2"]
        so2 = comp["so2"]
        co = comp["co"]
        o3 = comp["o3"]

        predicted_aqi = predict_aqi(
            model,
            pm25,
            pm10,
            no2,
            so2,
            co,
            o3
        )

        predictions.append(predicted_aqi)

        dates.append(pd.to_datetime(item["dt"], unit="s"))

    df = pd.DataFrame({
        "Date": dates,
        "Predicted AQI": predictions
    })

    st.line_chart(df.set_index("Date"))

    st.dataframe(df)