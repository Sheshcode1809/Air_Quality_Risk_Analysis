import requests

API_KEY = "395c63a31b5fc377f2d9458689350218"

def get_live_aqi(lat, lon):

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    response = requests.get(url)

    data = response.json()

    aqi = data["list"][0]["main"]["aqi"]

    pollutants = data["list"][0]["components"]

    return aqi, pollutants

def convert_aqi_scale(aqi):

    mapping = {
        1: 50,
        2: 100,
        3: 150,
        4: 200,
        5: 300
    }

    return mapping.get(aqi, 0)