import numpy as np

def predict_aqi(model, pm25, pm10, no2, so2, co, o3):
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(input_data)[0]
    return prediction