import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
body {
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background-color: #0F172A;  /* Dark Navy */
    color: white;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ---------- MAIN PANEL ---------- */
.stApp {
    background-color: #000000; /* Light Gray */
}

/* Remove default padding issues */
.block-container {
    padding: 2rem 2rem;
}

/* ---------- KPI CARDS ---------- */
.card {
    background-color: #0e1117;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #3B82F6;
}

/* ---------- TITLES ---------- */
h1, h2, h3 {
    color: #111827;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
}

/* ---------- METRIC COLORS ---------- */
.green {color: green;}
.orange {color: orange;}
.red {color: red;}

</style>
""", unsafe_allow_html=True)

from pathlib import Path

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "aqi_random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    model = None

# ------------------ RISK GUIDANCE ------------------
def get_risk_guidance(aqi):
    if aqi <= 50:
        return ["Air quality is good.", "Outdoor activities are safe."]
    elif aqi <= 100:
        return ["Air quality is moderate.", "Sensitive people take care."]
    elif aqi <= 200:
        return ["Reduce outdoor exposure."]
    elif aqi <= 300:
        return ["Wear mask outdoors."]
    else:
        return ["Stay indoors. Health emergency."]

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("Air Quality System")

    selected = option_menu(
        "Navigation",
        ["Dashboard", "Predict AQI", "Health Guide", "Action Hub","About"],
        icons=["house", "activity", "heart", "tree", "info-circle"],
        default_index=0,
    )

    st.sidebar.title("Project Info")
    st.sidebar.write("**Developed by:** Shesh Kanade")
    st.sidebar.write("**Status:** Prediction Engine v1.0")
    
@st.cache_data
def load_data():
    try:
        base_dir = Path(__file__).resolve().parent.parent
        csv_path = base_dir / "data" / "processed" / "air_quality_with_risk.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        data = pd.read_csv(csv_path)
        data['Datetime'] = pd.to_datetime(
            data["Datetime"],
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# ------------------ DASHBOARD ------------------
if selected == "Dashboard":
    st.set_page_config(page_title="Dashboard", layout="wide")
    st.title("Air Quality Risk Analysis Dashboard")
    st.markdown("Real-time Air Quality Monitoring & Insights Across India")
    
    # Add date range filter
    if df is not None and 'Datetime' in df.columns:
        min_date = df['Datetime'].min().date()
        max_date = df['Datetime'].max().date()
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter data by date range
        mask = (df['Datetime'].dt.date >= start_date) & (df['Datetime'].dt.date <= end_date)
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            st.warning("No data available for the selected date range. Showing all data.")
            df_filtered = df.copy()
    else:
        df_filtered = df.copy() if df is not None else None
    
    if df_filtered is not None and not df_filtered.empty:
        # 1. TOP-LEVEL METRICS
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        avg_aqi = df_filtered['AQI'].mean()
        max_aqi = df_filtered['AQI'].max()
        most_polluted_city = df_filtered.groupby('City')['AQI'].mean().idxmax()
        cleanest_city = df_filtered.groupby('City')['AQI'].mean().idxmin()
        
        # Calculate trend (compare last week vs previous week)
        if len(df_filtered) > 7:
            recent_avg = df_filtered.nlargest(7, 'Datetime')['AQI'].mean()
            previous_avg = df_filtered.nlargest(14, 'Datetime').nsmallest(7, 'Datetime')['AQI'].mean()
            aqi_trend = recent_avg - previous_avg
        else:
            aqi_trend = 0
        
        with col1:
            st.metric(
                label="National Average AQI", 
                value=f"{avg_aqi:.1f}", 
                delta=f"{aqi_trend:+.1f}",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Peak AQI Recorded", 
                value=f"{max_aqi:.0f}"
            )
        
        with col3:
            st.metric(
                label="Most Polluted City", 
                value=most_polluted_city
            )
        
        with col4:
            st.metric(
                label="Cleanest City", 
                value=cleanest_city
            )
        
        st.divider()
        
        # 2. MAIN CHARTS SECTION - Simplified Tabs
        tab1, tab2, tab3 = st.tabs(["Time Series Analysis", "City Rankings", "Pollutant Analysis"])
        
        with tab1:
            st.subheader("AQI Trends Over Time")
            
            # City selector for time series
            cities_for_trend = st.multiselect(
                "Select Cities to Compare",
                options=sorted(df_filtered['City'].unique()),
                default=[most_polluted_city, cleanest_city][:2] if len(df_filtered['City'].unique()) >= 2 else [most_polluted_city]
            )
            
            if cities_for_trend:
                fig_trend = go.Figure()
                
                # Add AQI category backgrounds
                fig_trend.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
                fig_trend.add_hrect(y0=51, y1=100, fillcolor="lightgreen", opacity=0.1, line_width=0)
                fig_trend.add_hrect(y0=101, y1=200, fillcolor="yellow", opacity=0.1, line_width=0)
                fig_trend.add_hrect(y0=201, y1=300, fillcolor="orange", opacity=0.1, line_width=0)
                fig_trend.add_hrect(y0=301, y1=400, fillcolor="red", opacity=0.1, line_width=0)
                fig_trend.add_hrect(y0=401, y1=500, fillcolor="darkred", opacity=0.1, line_width=0)
                
                # Add traces for selected cities
                colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6']
                for idx, city in enumerate(cities_for_trend):
                    city_data = df_filtered[df_filtered['City'] == city].sort_values('Datetime')
                    fig_trend.add_trace(go.Scatter(
                        x=city_data['Datetime'], 
                        y=city_data['AQI'],
                        mode='lines+markers', 
                        name=city,
                        line=dict(color=colors[idx % len(colors)], width=2),
                        marker=dict(size=4)
                    ))
                
                fig_trend.update_layout(
                    title="AQI Trends Comparison",
                    xaxis_title="Date",
                    yaxis_title="AQI Value",
                    template="plotly_dark",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Simple statistical summary
                st.markdown("Statistical Summary")
                summary_data = []
                for city in cities_for_trend:
                    city_data = df_filtered[df_filtered['City'] == city]['AQI']
                    summary_data.append({
                        'City': city,
                        'Average AQI': f"{city_data.mean():.1f}",
                        'Max AQI': f"{city_data.max():.0f}",
                        'Min AQI': f"{city_data.min():.0f}"
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            else:
                st.info("Please select at least one city to view trends")
        
        with tab2:
            st.subheader("City Performance Rankings")
            
            col_rank1, col_rank2 = st.columns(2)
            
            with col_rank1:
                # Most Polluted Cities
                city_avg_aqi = df_filtered.groupby('City')['AQI'].mean().round(1)
                city_avg_aqi = city_avg_aqi.sort_values(ascending=False)
                
                fig_worst = px.bar(
                    x=city_avg_aqi.head(10).values,
                    y=city_avg_aqi.head(10).index,
                    orientation='h',
                    title="Most Polluted Cities",
                    labels={'x': 'Average AQI', 'y': ''},
                    color=city_avg_aqi.head(10).values,
                    color_continuous_scale='Reds',
                    text=city_avg_aqi.head(10).values
                )
                fig_worst.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig_worst.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_worst, use_container_width=True)
            
            with col_rank2:
                # Cleanest Cities
                fig_best = px.bar(
                    x=city_avg_aqi.tail(10).values,
                    y=city_avg_aqi.tail(10).index,
                    orientation='h',
                    title="Cleanest Cities",
                    labels={'x': 'Average AQI', 'y': ''},
                    color=city_avg_aqi.tail(10).values,
                    color_continuous_scale='Greens',
                    text=city_avg_aqi.tail(10).values
                )
                fig_best.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig_best.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_best, use_container_width=True)
        
        with tab3:
            st.subheader("Pollutant Analysis")
            
            # Simple pollutant averages
            pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            if all(col in df_filtered.columns for col in pollutant_cols):
                
                # City selector for pollutant breakdown
                selected_city_pollutants = st.selectbox(
                    "Select City for Pollutant Details",
                    options=sorted(df_filtered['City'].unique())
                )
                
                city_pollutants = df_filtered[df_filtered['City'] == selected_city_pollutants][pollutant_cols].mean()
                
                # Pollutant bar chart
                fig_pollutants = px.bar(
                    x=city_pollutants.values,
                    y=city_pollutants.index,
                    orientation='h',
                    title=f"Average Pollutant Levels in {selected_city_pollutants}",
                    labels={'x': 'Concentration', 'y': 'Pollutant'},
                    color=city_pollutants.values,
                    color_continuous_scale='Viridis',
                    text=city_pollutants.values
                )
                fig_pollutants.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig_pollutants.update_layout(height=400)
                st.plotly_chart(fig_pollutants, use_container_width=True)
                
                # Simple correlation info
                st.markdown("Pollutant Correlation Information")
                corr_matrix = df_filtered[pollutant_cols].corr()
                
                # Show top correlations
                correlations = []
                for i in range(len(pollutant_cols)):
                    for j in range(i+1, len(pollutant_cols)):
                        correlations.append({
                            'Pollutant Pair': f"{pollutant_cols[i]} - {pollutant_cols[j]}",
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
                st.dataframe(corr_df.head(5), use_container_width=True)
        
        # 3. SIMPLE ALERTS SECTION
        st.divider()
        st.subheader("Air Quality Alerts")
        
        # Identify cities with critical AQI
        critical_cities = df_filtered[df_filtered['AQI'] > 300].groupby('City').size().reset_index(name='Critical Hours')
        if not critical_cities.empty:
            st.warning("Critical Alerts")
            for _, row in critical_cities.iterrows():
                st.write(f"- {row['City']}: {row['Critical Hours']} hours with AQI > 300")
            
            st.markdown("Recommended Actions:")
            st.markdown("- Stay indoors as much as possible")
            st.markdown("- Wear N95 masks when outdoors")
            st.markdown("- Keep windows and doors closed")
        else:
            st.success("No critical alerts at the moment")
        
        # 4. EXPORT OPTION
        st.divider()
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Dashboard Data (CSV)",
            data=csv_data,
            file_name=f"aqi_dashboard_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        st.error("Unable to load data. Please check the data file path and format.")
# ------------------ PREDICT AQI------------------
elif selected == "Predict AQI":
    st.set_page_config(page_title="AQI Predictor", layout="wide")
    st.title("Air Quality Predictor")
    st.markdown("Enter pollutant concentrations to get real-time AQI prediction")
    
    # Initialize session state variables if they don't exist
    if 'pm25_val' not in st.session_state:
        st.session_state.pm25_val = 0.0
    if 'pm10_val' not in st.session_state:
        st.session_state.pm10_val = 0.0
    if 'no2_val' not in st.session_state:
        st.session_state.no2_val = 0.0
    if 'so2_val' not in st.session_state:
        st.session_state.so2_val = 0.0
    if 'co_val' not in st.session_state:
        st.session_state.co_val = 0.0
    if 'o3_val' not in st.session_state:
        st.session_state.o3_val = 0.0
    if 'city_data_loaded' not in st.session_state:
        st.session_state.city_data_loaded = False
    
    # Get current values from session state for display
    current_pm25 = st.session_state.pm25_val
    current_pm10 = st.session_state.pm10_val
    current_no2 = st.session_state.no2_val
    current_so2 = st.session_state.so2_val
    current_co = st.session_state.co_val
    current_o3 = st.session_state.o3_val
    
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "City Selection"])
    
    with tab2:
        st.markdown("### Select City to Auto-fill Values")
        st.info("Select a city and click the button below to automatically fill all pollutant values in the Manual Input tab.")
        
        if df is not None:
            # Get unique cities
            cities = sorted(df['City'].unique())
            selected_city = st.selectbox("Choose a City", cities, key="city_selector")
            
            # Get latest data for selected city
            city_data = df[df['City'] == selected_city].sort_values('Datetime', ascending=False)
            if not city_data.empty:
                city_latest = city_data.iloc[0]
                
                # Display city stats in a nice format
                st.markdown("#### Current Pollutant Levels")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("PM2.5", f"{city_latest['PM2.5']:.1f} µg/m³")
                    st.metric("PM10", f"{city_latest['PM10']:.1f} µg/m³")
                with col_b:
                    st.metric("NO2", f"{city_latest['NO2']:.1f} µg/m³")
                    st.metric("SO2", f"{city_latest['SO2']:.1f} µg/m³")
                with col_c:
                    st.metric("CO", f"{city_latest['CO']:.1f} mg/m³")
                    st.metric("O3", f"{city_latest['O3']:.1f} µg/m³")
                
                # Button to populate values
                if st.button("Load This City's Data", use_container_width=True, key="load_city_data"):
                    # Update session state values
                    st.session_state.pm25_val = float(city_latest['PM2.5'])
                    st.session_state.pm10_val = float(city_latest['PM10'])
                    st.session_state.no2_val = float(city_latest['NO2'])
                    st.session_state.so2_val = float(city_latest['SO2'])
                    st.session_state.co_val = float(city_latest['CO'])
                    st.session_state.o3_val = float(city_latest['O3'])
                    st.session_state.city_data_loaded = True
                    st.success(f"Data from {selected_city} has been loaded! Switch to the Manual Input tab to view and edit the values.")
                    st.rerun()
            else:
                st.warning("No data available for this city.")
        else:
            st.error("Data not available. Please check the data file.")
     
    with tab1:
        st.markdown("### Enter Pollutant Values Manually")
        
        # FIXED ALIGNMENT - Using proper column structure with equal heights
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### Particulate Matter")
            
            pm25 = st.number_input(
                "PM2.5 (µg/m³)",
                min_value=0.0, 
                value=st.session_state.get("pm25_val", 0.0), 
                step=1.0,
                help="Fine particulate matter - most harmful to health",
                key="pm25_val"
            )
            # Update session state when user manually changes value
            if pm25 != st.session_state.pm25_val:
                st.session_state.pm25_val = pm25
                st.session_state.city_data_loaded = False
            
            if pm25 > 0:
                if pm25 <= 60:
                    st.caption("Good range")
                elif pm25 <= 120:
                    st.caption("Moderate range")
                else:
                    st.caption("Hazardous range")
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                
            pm10 = st.number_input(
                "PM10 (µg/m³)",
                min_value=0.0, 
                value=st.session_state.get("pm10_val", 0.0), 
                step=1.0,
                help="Coarse particulate matter",
                key="pm10_val"
            )
            if pm10 != st.session_state.pm10_val:
                st.session_state.pm10_val = pm10
                st.session_state.city_data_loaded = False
            
            if pm10 > 0:
                if pm10 <= 100:
                    st.caption("Good range")
                elif pm10 <= 250:
                    st.caption("Moderate range")
                else:
                    st.caption("Hazardous range")
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            so2 = st.number_input(
                "SO2 (µg/m³)",
                min_value=0.0, 
                value=st.session_state.get("so2_val", 0.0), 
                step=1.0,
                help="Sulfur dioxide - industrial emissions",
                key="so2_val"
            )
            if so2 != st.session_state.so2_val:
                st.session_state.so2_val = so2
                st.session_state.city_data_loaded = False
            
            if so2 > 0:
                if so2 <= 40:
                    st.caption("Good range")
                elif so2 <= 80:
                    st.caption("Moderate range")
                else:
                    st.caption("Hazardous range")
        
        with col2:
            st.markdown("#### Gaseous Pollutants")
            
            no2 = st.number_input(
                "NO2 (µg/m³)",
                min_value=0.0, 
                value=st.session_state.get("no2_val", 0.0), 
                step=1.0,
                help="Nitrogen dioxide - vehicle emissions",
                key="no2_val"
            )
            if no2 != st.session_state.no2_val:
                st.session_state.no2_val = no2
                st.session_state.city_data_loaded = False
            
            if no2 > 0:
                if no2 <= 40:
                    st.caption("Good range")
                elif no2 <= 80:
                    st.caption("Moderate range")
                else:
                    st.caption("Hazardous range")
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            co = st.number_input(
                "CO (mg/m³)",
                min_value=0.0, 
                value=st.session_state.get("co_val", 0.0), 
                step=0.1,
                help="Carbon monoxide - incomplete combustion",
                key="co_val"
            )
            if co != st.session_state.co_val:
                st.session_state.co_val = co
                st.session_state.city_data_loaded = False
            
            if co > 0:
                if co <= 1.0:
                    st.caption("Good range")
                elif co <= 2.0:
                    st.caption("Moderate range")
                else:
                    st.caption("Hazardous range")
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            o3 = st.number_input(
                "O3 (µg/m³)",
                min_value=0.0, 
                value=st.session_state.get("o3_val", 0.0), 
                step=1.0,
                help="Ozone - secondary pollutant",
                key="o3_val"
            )
            if o3 != st.session_state.o3_val:
                st.session_state.o3_val = o3
                st.session_state.city_data_loaded = False
            
            if o3 > 0:
                if o3 <= 50:
                    st.caption("Good range")
                elif o3 <= 100:
                    st.caption("Moderate range")
                else:
                    st.caption("Hazardous range")
        
        # Show a success message if city data was loaded
        if st.session_state.city_data_loaded:
            st.success("City data has been loaded successfully! All 6 pollutant values are now filled. You can edit them manually if needed.")
        # Divider and prediction section
        st.divider()
        
        # Get current values from session state (these will reflect both manual input and city-loaded data)
        pm25 = st.session_state.pm25_val
        pm10 = st.session_state.pm10_val
        no2 = st.session_state.no2_val
        so2 = st.session_state.so2_val
        co = st.session_state.co_val
        o3 = st.session_state.o3_val
        
        # Check if all inputs are provided
        inputs = [pm25, pm10, no2, so2, co, o3]
        all_inputs_provided = all(v > 0 for v in inputs)
        
        # Display progress indicator
        filled_inputs = sum(1 for v in inputs if v > 0)
        st.write(f"**Input Status:** {filled_inputs}/6 pollutants entered")
        
        if filled_inputs == 0:
            st.info("Please enter pollutant values manually in the Manual Input tab or use the City Selection tab to load data.")
        elif filled_inputs > 0 and filled_inputs < 6:
            missing = []
            if pm25 == 0: missing.append("PM2.5")
            if pm10 == 0: missing.append("PM10")
            if no2 == 0: missing.append("NO2")
            if so2 == 0: missing.append("SO2")
            if co == 0: missing.append("CO")
            if o3 == 0: missing.append("O3")
            st.warning(f"Please enter all 6 pollutant values. Missing: {', '.join(missing)}")
        elif all_inputs_provided:
            st.success("All 6 pollutant values have been entered! You can now click 'Predict AQI'.")
    
    st.divider()
    
    predict_button = st.button("Predict AQI", use_container_width=True)
    
    # Prediction button and results
    col_pred, col_insight = st.columns([1,1])
       
    with col_pred:
        
        if predict_button and all_inputs_provided:
            try:
                pred = model.predict([inputs])[0] if model else 125.0
                
                # Save to session state
                st.session_state['aqi_result'] = pred
                st.session_state['pollutants_used'] = inputs
                st.session_state['prediction_made'] = True
                st.session_state['prediction_timestamp'] = pd.Timestamp.now()
                st.session_state['last_prediction'] = {
                    'aqi': pred,
                    'inputs': inputs.copy(),
                    'timestamp': pd.Timestamp.now()
                }
                
                if pred <= 50:
                    aqi_color = "green"
                    aqi_category = "Good"
                elif pred <= 100:
                    aqi_color = "lightgreen"
                    aqi_category = "Satisfactory"
                elif pred <= 200:
                    aqi_color = "orange"
                    aqi_category = "Moderate"
                elif pred <= 300:
                    aqi_color = "#FF6600"
                    aqi_category = "Poor"
                elif pred <= 400:
                    aqi_color = "red"
                    aqi_category = "Very Poor"
                else:
                    aqi_color = "darkred"
                    aqi_category = "Severe"
                    
                # Display Predicted AQI card
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background-color: #1e1e1e; margin: 10px 0; height: 100%;">
                <h3 style="color: #cccccc; margin-bottom: 10px;">Predicted AQI Value</h3>
                <h1 style="color: {aqi_color}; font-size: 72px; font-weight: bold; margin: 20px 0;">{pred:.1f}</h1>
                <h2 style="color: {aqi_color}; margin: 10px 0;">Category: {aqi_category}</h2>
                </div>
                """, unsafe_allow_html=True)                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    with col_insight:
        if predict_button and all_inputs_provided:
            # Pollutant Contribution Pie Chart
            pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            values = inputs
                    
            total = sum(values)
            if total > 0:
                percentages = [v/total*100 for v in values]
            fig_pie = px.pie(
            values=percentages, 
            names=pollutants,
            title="Pollutant Contribution",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            hole=0.3  # Adds a donut hole for better look
            )
            fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12
            )
            fig_pie.update_layout(
            height=380,
            margin=dict(t=40, b=0, l=0, r=0),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Check if there's a prediction to display
        display_prediction = False
        aqi = None
        inputs_display = None
        
        if 'last_prediction' in st.session_state and st.session_state.get('prediction_made', False):
            display_prediction = True
            aqi = st.session_state['last_prediction']['aqi']
            inputs_display = st.session_state['last_prediction']['inputs']
        elif 'aqi_result' in st.session_state and st.session_state.get('prediction_made', False):
            display_prediction = True
            aqi = st.session_state['aqi_result']
            inputs_display = [pm25, pm10, no2, so2, co, o3]
        
        if display_prediction:
            # Determine category
            if aqi <= 50:
                category, color = "Good", "green"
            elif aqi <= 100:
                category, color = "Satisfactory", "lightgreen"
            elif aqi <= 200:
                category, color = "Moderate", "yellow"
            elif aqi <= 300:
                category, color = "Poor", "orange"
            elif aqi <= 400:
                category, color = "Very Poor", "red"
            else:
                category, color = "Severe", "darkred"
        
    # Add a helpful note linking to Health Guide
    if 'aqi_result' in st.session_state and st.session_state.get('prediction_made', False):
        st.info("Need health advice? Go to the Health Guide page for detailed recommendations based on this AQI value.")
    
    # Add historical comparison if data available and prediction made
    if df is not None and 'aqi_result' in st.session_state and st.session_state.get('prediction_made', False):
        st.divider()
        st.subheader("Compare with Historical Data")
        
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            historical_aqi = df['AQI'].dropna()
            pred_aqi = st.session_state['aqi_result']
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=historical_aqi, nbinsx=30, name="Historical AQI"))
            fig_hist.add_vline(x=pred_aqi, line_dash="dash", line_color="red", 
                              annotation_text=f"Your Prediction: {pred_aqi:.1f}")
            fig_hist.update_layout(
                title="Your Prediction vs Historical Distribution",
                xaxis_title="AQI Value",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_hist2:
            city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(5)
            fig_bar = px.bar(
                x=city_avg.values, 
                y=city_avg.index,
                orientation='h',
                title="Most Polluted Cities (Historical Avg)",
                labels={'x': 'Average AQI', 'y': 'City'},
                color=city_avg.values,
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
        
    # Add explanatory tooltips and tips
    with st.expander("Understanding Pollutant Levels"):
        st.markdown("""
        ### Pollutant Reference Ranges (24-hour average)
        
        | Pollutant | Good | Moderate | Unhealthy | Hazardous |
        |-----------|------|----------|-----------|-----------|
        | **PM2.5** | 0-30 | 31-60 | 61-90 | >90 |
        | **PM10** | 0-50 | 51-100 | 101-250 | >250 |
        | **NO2** | 0-40 | 41-80 | 81-180 | >180 |
        | **SO2** | 0-40 | 41-80 | 81-380 | >380 |
        | **CO** | 0-1.0 | 1.1-2.0 | 2.1-10 | >10 |
        | **O3** | 0-50 | 51-100 | 101-168 | >168 |
        
        """)
# ------------------ HEALTH GUIDE------------------
elif selected == "Health Guide":
    st.set_page_config(page_title="Health Guide", layout="wide")
    st.title("Health Guide and Precautions")
    st.markdown("Comprehensive health recommendations based on AQI levels and individual risk factors")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["AQI-Based Advice", "Risk Groups", "Protection Tips"])
    
    with tab1:
        st.subheader("AQI Level-Based Health Recommendations")
        
        # Show current prediction if available
        if 'aqi_result' in st.session_state:
            aqi = st.session_state['aqi_result']
            prediction_time = st.session_state.get('prediction_timestamp', pd.Timestamp.now())
            st.info(f"Current Predicted AQI: {aqi:.2f} (Predicted on: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')})")

            
            # Detailed AQI category with comprehensive advice
            if aqi <= 50:
                st.success("### AQI Category: GOOD (0-50)")
                st.markdown("""
                **Air Quality:** Excellent air quality with minimal pollution.
                
                **Health Impact:** 
                - No health risks for anyone
                - Perfect for outdoor activities
                - Great time for exercise and sports
                
                **Recommendations:**
                - Enjoy outdoor activities freely
                - Open windows for ventilation
                - Perfect for morning walks and exercise
                - Great time for outdoor gatherings
                
                **Best Activities:**
                - Running, jogging, or cycling
                - Yoga in parks
                - Outdoor sports
                - Long walks with family
                """)
                
            elif aqi <= 100:
                st.warning("### AQI Category: SATISFACTORY (51-100)")
                st.markdown("""
                **Air Quality:** Acceptable air quality with some pollution concerns.
                
                **Health Impact:** 
                - Minimal health impacts for general population
                - Very sensitive individuals may experience minor discomfort
                
                **Recommendations:**
                - Normal outdoor activities can continue
                - Sensitive individuals should monitor symptoms
                - Good for moderate outdoor exercise
                - Keep windows open for ventilation
                
                **For Sensitive Groups:**
                - Limit prolonged outdoor exertion
                - Keep rescue inhalers handy if asthmatic
                - Monitor for any breathing difficulties
                """)
                
            elif aqi <= 200:
                st.warning("### AQI Category: MODERATE (101-200)")
                st.markdown("""
                **Air Quality:** Unhealthy for sensitive groups.
                
                **Health Impact:** 
                - People with respiratory or heart conditions may experience discomfort
                - General population may have minor symptoms
                
                **Recommendations:**
                - Consider wearing masks outdoors
                - Limit outdoor activities to morning hours
                - Keep windows closed during peak pollution hours
                - Use air purifiers indoors
                
                **Specific Advice:**
                - **Children and Elderly:** Limit outdoor play/activities
                - **Asthma Patients:** Keep medication handy, avoid strenuous activities
                - **Heart Patients:** Monitor symptoms, avoid morning walks
                - **Pregnant Women:** Minimize outdoor exposure
                """)
                
            elif aqi <= 300:
                st.error("### AQI Category: POOR (201-300)")
                st.markdown("""
                **Air Quality:** Unhealthy for everyone.
                
                **Health Impact:** 
                - Everyone may experience health effects
                - Serious health concerns for sensitive groups
                - Visible haze likely
                
                **Recommendations:**
                - **Stay indoors as much as possible**
                - **Wear N95 masks when outdoors**
                - Avoid all outdoor physical activities
                - Keep doors and windows closed
                - Use HEPA air purifiers
                - Use humidifiers to reduce irritation
                
                **Medical Precautions:**
                - **Asthma:** Use preventive inhalers, stay indoors
                - **Heart Conditions:** Avoid exertion, monitor symptoms
                - **Elderly:** Stay indoors, maintain medication schedule
                - **Children:** Indoor activities only, use air purifiers in rooms
                
                **If You Must Go Out:**
                - Wear properly fitted N95 mask
                - Limit outdoor time to under 30 minutes
                - Avoid busy roads and traffic areas
                - Shower immediately after returning
                """)
                
            elif aqi <= 400:
                st.error("### AQI Category: VERY POOR (301-400)")
                st.markdown("""
                **Air Quality:** Serious health emergency conditions.
                
                **Health Impact:** 
                - Significant health risks for everyone
                - Severe symptoms in sensitive groups
                - Medical attention may be required
                
                **Critical Recommendations:**
                - **STAY INDOORS - DO NOT GO OUTSIDE UNLESS ABSOLUTELY NECESSARY**
                - **Must wear N95 masks even indoors if no air purifier**
                - **No outdoor activities whatsoever**
                - Seal windows and doors properly
                - Run air purifiers 24/7
                - Keep all medications accessible
                
                **Medical Emergency Signs:**
                - Difficulty breathing
                - Chest pain or tightness
                - Severe coughing fits
                - Dizziness or confusion
                
                **Immediate Actions:**
                1. Close all windows and doors
                2. Turn on air purifiers
                3. Stay in one room to minimize exposure
                4. Contact doctor if symptoms worsen
                5. Avoid using vacuum cleaners
                """)
                
            else:
                st.error("### AQI Category: SEVERE (401-500)")
                st.markdown("""
                **Air Quality:** Hazardous emergency conditions.
                
                **Health Impact:** 
                - **EMERGENCY SITUATION**
                - Serious health risks for entire population
                - Immediate medical attention for symptoms
                - Life-threatening for sensitive groups
                
                **EMERGENCY RECOMMENDATIONS:**
                - **DO NOT GO OUTDOORS UNDER ANY CIRCUMSTANCES**
                - **Seek medical attention if experiencing symptoms**
                - **Create a clean room with sealed windows**
                - **Use multiple air purifiers if available**
                - **Keep emergency medications ready**
                - **Keep emergency contacts accessible**
                
                **Warning Signs Requiring Medical Help:**
                - Severe breathing difficulty
                - Chest pain or pressure
                - Blue lips or face
                - Confusion or fainting
                - Severe headache
                
                **For Sensitive Groups (Children, Elderly, Pregnant):**
                - Monitor closely for any symptoms
                - Keep in rooms with air purifiers
                - Avoid any physical activity
                - Ensure proper hydration
                - Have emergency numbers ready
                
                **Preparedness:**
                - Stock up on essential medications
                - Have N95 masks available
                - Keep emergency contact numbers handy
                - Monitor official air quality alerts
                """)
            
            # Add a clear button to reset prediction
            st.divider()
            if st.button("Clear Prediction Data", use_container_width=True):
                if 'aqi_result' in st.session_state:
                    del st.session_state['aqi_result']
                if 'prediction_made' in st.session_state:
                    del st.session_state['prediction_made']
                if 'pollutants_used' in st.session_state:
                    del st.session_state['pollutants_used']
                st.success("Prediction data cleared. You can now make a new prediction.")
                st.rerun()
        
        else:
            st.info("No prediction data found. Please run a prediction on the Predict AQI page first to get personalized health advice.")
            
            
            # Show sample advice for demonstration
            st.markdown("### Sample Health Guide (Example)")
            with st.expander("View Sample Recommendations for Different AQI Levels"):
                st.markdown("""
                **Good AQI (0-50):** Perfect for outdoor activities, no restrictions.
                
                **Satisfactory AQI (51-100):** Normal activities can continue; sensitive individuals should monitor symptoms.
                
                **Moderate AQI (101-200):** Sensitive groups should limit outdoor exposure, others can continue normal activities with caution.
                
                **Poor AQI (201-300):** Everyone should reduce outdoor activities, wear masks when outside.
                
                **Very Poor AQI (301-400):** Stay indoors, avoid all outdoor activities, use air purifiers.
                
                **Severe AQI (401-500):** Emergency conditions, stay indoors at all costs, seek medical help if symptomatic.
                """)
    
    with tab2:
        st.subheader("Risk Group-Specific Recommendations")
        
        # Create expandable sections for different risk groups
        with st.expander("Elderly (Age 65+)", expanded=True):
            st.markdown("""
            **Why at Risk:** 
            - Weakened immune system
            - Pre-existing conditions (heart disease, diabetes, respiratory issues)
            - Reduced lung capacity
            
            **Protection Measures:**
            - Stay indoors during high pollution days
            - Take medications as prescribed
            - Regular check-ins with family/caregivers
            - Avoid morning walks when AQI > 100
            - Use N95 masks for essential outdoor trips
            - Keep indoor air clean with purifiers
            
            **Symptoms to Watch:**
            - Shortness of breath
            - Chest discomfort
            - Excessive coughing
            - Fatigue
            - Confusion
            """)
        
        with st.expander("Children and Infants"):
            st.markdown("""
            **Why at Risk:** 
            - Developing lungs and immune systems
            - Higher breathing rate per body weight
            - More time spent outdoors playing
            
            **Protection Measures:**
            - Limit outdoor play when AQI > 150
            - Encourage indoor activities and games
            - Schools should keep children indoors during poor AQI
            - Teach proper mask usage for older children
            - Use air purifiers in bedrooms
            - Use car cabin air filters
            
            **For Infants:**
            - Keep indoors during high pollution
            - Use stroller covers with filters
            - Breastfeed to boost immunity
            - Regular pediatric check-ups
            """)
        
        with st.expander("Pregnant Women"):
            st.markdown("""
            **Why at Risk:** 
            - Air pollution can affect fetal development
            - Increased risk of preterm birth
            - Potential developmental issues
            
            **Protection Measures:**
            - Avoid outdoor activities when AQI > 150
            - Regular prenatal check-ups
            - Maintain clean indoor environment
            - Eat antioxidant-rich foods
            - Stay well hydrated
            - N95 masks for essential outdoor trips
            
            **Nutritional Support:**
            - Vitamin C rich foods (oranges, kiwis)
            - Omega-3 fatty acids (fish, nuts)
            - Green leafy vegetables
            - Adequate water intake
            """)
        
        with st.expander("Asthma and Respiratory Patients"):
            st.markdown("""
            **Why at Risk:** 
            - Direct trigger for asthma attacks
            - Inflammation of airways
            - Reduced lung function
            
            **Protection Measures:**
            - Keep rescue inhalers accessible at all times
            - Stay indoors when AQI > 100
            - Follow asthma action plan
            - Use HEPA air purifiers
            - Avoid known triggers (smoke, strong odors)
            - Wear mask even indoors if needed
            
            **Emergency Plan:**
            1. Use rescue inhaler at first sign of symptoms
            2. Move to clean indoor air
            3. Use peak flow meter to monitor
            4. Seek medical help if symptoms don't improve
            5. Keep emergency contacts ready
            """)
        
        with st.expander("Heart Disease Patients"):
            st.markdown("""
            **Why at Risk:** 
            - Increased risk of heart attacks
            - Blood pressure fluctuations
            - Inflammation of blood vessels
            
            **Protection Measures:**
            - Take medications on schedule
            - Monitor blood pressure regularly
            - Avoid strenuous activities in poor air
            - No morning walks when AQI > 150
            - Use N95 masks for essential trips
            - Keep cardiologist contact handy
            
            **Warning Signs:**
            - Chest pain or pressure
            - Irregular heartbeat
            - Shortness of breath
            - Excessive fatigue
            - Dizziness
            """)
    
    with tab3:
        st.subheader("Protection Tips and Best Practices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Indoor Protection")
            st.markdown("""
            **Air Purification:**
            - Use HEPA/activated carbon air purifiers
            - Place purifiers in bedrooms and living areas
            - Change filters regularly (every 3-6 months)
            - Consider indoor plants: Areca Palm, Snake Plant, Spider Plant
            
            **Ventilation Strategy:**
            - Keep windows closed during peak pollution hours (6-10 AM, 5-8 PM)
            - Use exhaust fans in kitchens and bathrooms
            - Ventilate during early morning if AQI is good
            - Seal gaps around doors and windows
            
            **Cleaning Practices:**
            - Use vacuum cleaners with HEPA filters
            - Wet mop floors instead of dry sweeping
            - Avoid burning incense or candles
            - Don't use chemical air fresheners
            """)
            
            st.markdown("### Diet and Nutrition")
            st.markdown("""
            **Foods That Help:**
            - **Vitamin C:** Oranges, lemons, kiwis, bell peppers
            - **Omega-3:** Fish, walnuts, flaxseeds
            - **Antioxidants:** Berries, turmeric, green tea
            - **Magnesium:** Nuts, seeds, leafy greens
            
            **Hydration:**
            - Drink 8-10 glasses of water daily
            - Herbal teas (green tea, ginger tea)
            - Avoid caffeine and alcohol
            """)
        
        with col2:
            st.markdown("### Outdoor Protection")
            st.markdown("""
            **Mask Selection:**
            - **N95/N99:** Best protection for pollution
            - **Surgical masks:** Minimal protection
            - **Cloth masks:** Not effective for PM2.5
            
            **Proper Mask Usage:**
            - Ensure tight seal around nose and mouth
            - Replace when damp or after 8 hours of use
            - Don't reuse disposable masks
            - Fit is critical - one size doesn't fit all
            
            **Timing Matters:**
            - Best time: Early morning (4-6 AM) or after rain
            - Avoid: Rush hours (8-10 AM, 5-8 PM)
            - Check AQI before planning outdoor activities
            
            **Activities to Avoid:**
            - Jogging on busy roads
            - Outdoor sports during high AQI
            - Burning waste or leaves
            - Using diesel generators near living spaces
            """)
            
            st.markdown("### Travel Tips")
            st.markdown("""
            **Driving:**
            - Keep windows rolled up in traffic
            - Use recirculation mode in AC
            - Check vehicle PUC certificate regularly
            - Consider carpooling to reduce emissions
            
            **Public Transport:**
            - Prefer metro/metro over buses
            - Wear masks in crowded stations
            - Avoid peak hours if possible
            """)
        
        # Add downloadable guide
        st.divider()
        st.subheader("Download Complete Health Guide")
        
        # Create comprehensive guide for download
        guide_content = """Air Quality Health Guide
================================

AQI Categories and Recommendations:

1. GOOD (0-50)
- No restrictions, enjoy outdoor activities

2. SATISFACTORY (51-100)
- Normal activities continue
- Sensitive groups should monitor symptoms

3. MODERATE (101-200)
- Limit outdoor activities
- Wear masks if sensitive
- Keep windows closed during peak hours

4. POOR (201-300)
- Stay indoors
- Wear N95 masks
- Avoid outdoor exercise

5. VERY POOR (301-400)
- Emergency conditions
- Do not go outdoors
- Use air purifiers

6. SEVERE (401-500)
- Hazardous conditions
- Stay in clean rooms
- Seek medical help if symptomatic

General Protection Tips:
- Use HEPA air purifiers
- Wear N95 masks outdoors
- Keep windows closed
- Stay hydrated
- Eat antioxidant-rich foods
- Monitor AQI regularly

Risk Groups:
- Elderly, Children, Pregnant women
- Asthma and heart patients
- Take extra precautions

For more information, consult your healthcare provider.
"""
        
        st.download_button(
            label="Download Complete Health Guide (TXT)",
            data=guide_content,
            file_name="AQI_Health_Guide.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.caption("Note: These recommendations are based on CPCB guidelines and WHO air quality standards. Always consult your healthcare provider for personalized medical advice.")

# ------------------ SUSTAINABILITY & ACTION HUB ------------------
elif selected == "Action Hub":
    st.set_page_config(page_title="Action Hub", layout = "wide")
    st.title(" Sustainability & Action Hub")
    st.markdown("Your choices today determine the air we breathe tomorrow. Here’s how you can help.")

    # --- SECTION 1: PERSONAL ACTION CHECKLIST ---
    st.subheader(" User Daily Green Checklist")
    col1, col2 = st.columns(2)
    
    with col1:
        used_public_transport = st.checkbox("Used public transport/Carpooling")
        no_waste_burning = st.checkbox("Avoided burning dry leaves/trash")
        planted_trees = st.checkbox("Planted a sapling or tended to indoor plants")
    with col2:
        energy_saved = st.checkbox("Switched off lights/AC when not in use")
        checked_puc = st.checkbox("Vehicle PUC (Pollution Under Control) is valid")
        spread_awareness = st.checkbox("Talked to someone about air quality")

    # Interactive progress bar based on checklist
    actions = [used_public_transport, no_waste_burning, planted_trees, energy_saved, checked_puc, spread_awareness]
    score = sum(actions)
    progress = score / len(actions)
    
    st.write(f"**Your Impact Score for Today:** {score}/{len(actions)}")
    st.progress(progress)
    
    if score == len(actions):
        st.success(" Eco-Warrior Status! You're making a massive difference!")

    st.divider()
    # --- SECTION 2: REDUCE EXPOSURE VS REDUCE EMISSION ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Reduce Your Exposure")
        st.info("""
        - **Check AQI daily** before heading out.
        - **Use N95 Masks** when AQI is above 200.
        - **Indoor Plants:** Keep Areca Palms or Aloe Vera to improve indoor air.
        - **Air Purifiers:** Use HEPA filters during 'Severe' days.
        """)

    with col_b:
        st.markdown("### Reduce Your Emissions")
        st.warning("""
        - **Engine Off:** Turn off your vehicle at red lights (Idling kills!).
        - **E-Vehicles:** Consider switching to Electric Vehicles (EVs).
        - **LPG over Wood:** Use clean cooking fuels to reduce indoor smoke.
        - **Report:** Use the **Sameer App** to report illegal garbage burning.
        """)

    # --- SECTION 3: EDUCATIONAL RESOURCES ---
    st.divider()
    st.subheader("Learn More")
    expander = st.expander("Why does PM2.5 matter?")
    expander.write("""
    PM2.5 are tiny particles (2.5 microns or less) that can enter deep into the lungs and even the bloodstream. 
    Reducing these through green energy and better waste management is the top priority for public health in India.
    """)
    
    st.caption("Data and guidelines sourced from National Clean Air Programme (NCAP).")

# ------------------ ABOUT ------------------
else:
    st.set_page_config(page_title="About", layout = "wide")
    # Title and Introduction
    st.title("Air Quality Risk Analysis")
    st.markdown("""
    ### **The Problem**
    Rapid urbanization and industrial growth have made air pollution a serious public health concern. 
    Most current monitoring systems only report air quality *after* exposure has already occurred.
    """)
    
    # Highlighting the Solution
    st.info("**Our Project:**uses machine learning to provide real-time air quality predictions.It aims to help users make informed decisions about their outdoor activities based on pollutant levels.")

    # Project Pillars in Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Data Sources")
        st.write("We integrate data from high-authority sources to ensure model reliability:")
        st.markdown("- **OpenAQ:** For global harmonized air quality data.")
        st.markdown("- **CPCB:** For official Indian regulatory pollutants.")
        st.markdown("- **Meteorological Factors:** Temperature, Wind Speed, and Humidity.")

    with col2:
        st.subheader(" Machine Learning Models")
        st.write("Our system compares two distinct approaches:")
        st.markdown("- **Random Forest:** Handles non-linear relationships for robust risk classification.")
        st.markdown("- **LSTM (Long Short-Term Memory):** A Deep Learning model specifically designed for complex time-series forecasting.")

    st.divider()

    # AQI Standards Reference Table
    st.subheader("National AQI Standards (India)")
    aqi_data = {
        "AQI Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        "Range": ["0-50", "51-100", "101-200", "201-300", "301-400", "401-500"],
        "Health Impact": ["Minimal", "Minor discomfort", "Breathing discomfort", "Heart/Lung issues", "Respiratory illness", "Serious impacts"]
    }
    
    st.table(pd.DataFrame(aqi_data))

    st.markdown("""
                
        **AQI PREDICTOR USER MANUAL**
        
        HOW TO USE THIS WEBSITE:
        1. Navigate to the 'Predict AQI' tab.
        2. Enter the values for pollutants (PM2.5, PM10, NO2, etc.).
        3. Click 'Predict'—if inputs are empty, you will see a warning animation.
        4. View your result and visit the 'Health Guide' for personalized advice.
    
        TARGET AUDIENCE:
        - Health-conscious individuals and families.
        - Vulnerable groups (children, elderly, and those with respiratory issues).
        - Outdoor athletes and event planners.
        - Environmental researchers and urban planners.
    """
    )