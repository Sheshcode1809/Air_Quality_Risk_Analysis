# Air Quality Risk Analysis

## Overview

This project is a comprehensive air quality monitoring and risk assessment system built with Python and Streamlit. It analyzes air quality data from various Indian cities, predicts Air Quality Index (AQI) using machine learning models, and provides health guidance and actionable recommendations based on air quality levels.

## Features

- **Interactive Dashboard**: Visualize air quality trends and statistics across different cities
- **AQI Prediction**: Predict air quality index using a trained Random Forest model
- **Health Guide**: Get personalized health recommendations based on AQI levels
- **Action Hub**: Access resources and tips for improving air quality
- **Data Exploration**: Jupyter notebooks for comprehensive data analysis and visualization

## Technologies Used

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning for AQI prediction
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive charts and graphs
- **Joblib** - Model serialization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sheshcode1809/Air_Quality_Risk_Analysis.git
   cd Air_Quality_Risk_Analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app/app.py
   ```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the sidebar navigation to explore different sections:
   - **Dashboard**: View air quality statistics and trends
   - **Predict AQI**: Input parameters to predict air quality index
   - **Health Guide**: Get health recommendations based on AQI
   - **Action Hub**: Access air quality improvement resources
   - **About**: Learn more about the project

## Project Structure

```
Air_Quality_Risk_Analysis/
├── app/
│   └── app.py                 # Main Streamlit application
├── data/
│   ├── raw/                   # Raw air quality datasets
│   │   ├── city_day.csv
│   │   ├── city_hour.csv
│   │   ├── station_day.csv
│   │   ├── station_hour.csv
│   │   └── stations.csv
│   └── processed/             # Cleaned and processed data
│       ├── air_quality_cleaned.csv
│       └── air_quality_with_risk.csv
├── models/
│   └── aqi_random_forest.pkl  # Trained Random Forest model
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Initial data exploration
│   ├── 02_data_preprocessing.ipynb  # Data cleaning and preprocessing
│   ├── 03_EDA_visualization.ipynb   # Exploratory data analysis
│   └── 04_model_training.ipynb      # Model development and training
├── utils/                    # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Data Sources

The project uses air quality data from various monitoring stations across Indian cities. The raw data includes:

- **City-level daily and hourly data**: Aggregated air quality measurements
- **Station-level data**: Detailed measurements from individual monitoring stations
- **Station information**: Location and metadata for monitoring stations

Data features include PM2.5, PM10, NO2, SO2, CO, O3 concentrations, and calculated AQI values.

## Notebooks

1. **01_data_exploration.ipynb**: Initial data loading, basic statistics, and data understanding
2. **02_data_preprocessing.ipynb**: Data cleaning, handling missing values, and feature engineering
3. **03_EDA_visualization.ipynb**: Exploratory data analysis with visualizations and insights
4. **04_model_training.ipynb**: Machine learning model development, training, and evaluation

## Model

The AQI prediction model is a Random Forest regressor trained on historical air quality data. It predicts AQI values based on pollutant concentrations and other environmental factors.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Air quality data sourced from public environmental monitoring datasets
- Built with Streamlit for interactive data applications
- Machine learning implementation using scikit-learn