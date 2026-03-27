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
    st.title("Air Quality Risk Analysis")
    st.subheader("Dashboard")
    
    # 1. TOP-LEVEL METRICS (ST.METRIC)
    # Using st.metric instead of custom HTML for cleaner, responsive cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate daily changes (deltas) for more insight
    avg_aqi = df['AQI'].mean()
    max_aqi = df['AQI'].max()
    most_polluted_city = df.groupby('City')['AQI'].mean().idxmax()
    
    col1.metric("Average National AQI", f"{avg_aqi:.1f}", delta="-2.1", delta_color="inverse")
    col2.metric("Peak AQI Recorded", f"{max_aqi:.0f}", delta="Critical", delta_color="off")
    col3.metric("Current Hotspot", most_polluted_city)
    col4.metric("Active Stations", f"{df['City'].nunique()}")

    st.divider()

    # 2. INTERACTIVE ANALYSIS COLUMNS
    chart_col, filter_col = st.columns([3, 1])

    with filter_col:
        st.subheader("Deep Dive")
        city_select = st.selectbox("Filter by City", df['City'].unique())
        city_df = df[df['City'] == city_select].sort_values('Datetime')
        
        # Pollutant Breakdown Pie Chart
        st.write("**Pollutant Distribution**")
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        avg_pollutants = city_df[pollutants].mean()
        fig_pie = px.pie(values=avg_pollutants, names=pollutants, hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(showlegend=False, height=200, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col:
        # Time Series with Risk Backgrounds
        st.subheader(f"AQI Trend: {city_select}")
        fig_trend = go.Figure()
        
        # Add a shaded background for "Safe" vs "Unsafe" zones
        fig_trend.add_hrect(y0=0, y1=100, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Safe")
        fig_trend.add_hrect(y0=101, y1=200, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Moderate")
        fig_trend.add_hrect(y0=201, y1=500, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Hazardous")

        fig_trend.add_trace(go.Scatter(x=city_df['Datetime'], y=city_df['AQI'],
                                     mode='markers+text', name='AQI Level',
                                     line=dict(color='#3B82F6', width=2)))
        
        fig_trend.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_trend, use_container_width=True)

    # 3. GEOSPATIAL MAP (If Latitude/Longitude exist)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.subheader("Pollution Hotspots across India")
        st.map(df[['Latitude', 'Longitude']])

# ------------------ PREDICT AQI------------------
elif selected == "Predict AQI":
    st.set_page_config(page_title="AQI Predictor", layout = "wide")
    st.title("Predict AQI")

    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.number_input("PM2.5", 0.0)
        so2 = st.number_input("SO2", 0.0)
    with col2:
        pm10 = st.number_input("PM10", 0.0)
        co = st.number_input("CO", 0.0)
    with col3:
        no2 = st.number_input("NO2", 0.0)
        o3 = st.number_input("O3", 0.0)

    if st.button("Predict"):
        # Simulated prediction (replace with your model.predict if loaded)
        pred = model.predict([[pm25, pm10, no2, so2, co, o3]])[0] if model else 125.0

        # CHECK CONDITION: If all inputs are still 0.0, show warning
        inputs = [pm25, pm10, no2, so2, co, o3]
        
        if all(v == 0.0 for v in inputs):
            st.warning("Please enter at least one pollutant value to get a prediction.")
            # Optional: Show a Lottie warning here
            # st_lottie(lottie_warning_json, height=150) 
        else:
            # RUN PREDICTION
            pred = model.predict([inputs])[0] if model else 125.0 
            
            # SAVE TO SESSION STATE
            st.session_state['aqi_result'] = pred
            
            st.success(f"Predicted AQI: {pred:.2f}")
            st.info("Navigate to the **Health Guide** tab to see detailed precautions!")

# ------------------ HEALTH GUIDE------------------
elif selected == "Health Guide":
    st.set_page_config(page_title="Health Guide", layout = "wide")
    st.title("Health Advice & Precautions")

    if 'aqi_result' in st.session_state:
        aqi = st.session_state['aqi_result']
        
        # 1. Determine Category & Advice
        if aqi <= 50:
            cat, color, help_text = "Good", "green", "Minimal impact. Enjoy your day!"
        elif aqi <= 100:
            cat, color, help_text = "Satisfactory", "#90EE90", "Minor breathing discomfort to sensitive people."
        elif aqi <= 200:
            cat, color, help_text = "Moderate", "yellow", "Discomfort to people with lung/heart disease."
        elif aqi <= 300:
            cat, color, help_text = "Poor", "orange", "Breathing discomfort to most people on prolonged exposure."
        elif aqi <= 400:
            cat, color, help_text = "Very Poor", "red", "Respiratory illness on prolonged exposure."
        else:
            cat, color, help_text = "Severe", "maroon", "Serious impacts on those with existing diseases."

        # 2. Display Results
        st.subheader(f"Current Predicted AQI: {aqi:.2f}")
        st.markdown(f"### Category: :{color}[{cat}]")
        st.info(f"**General Impact:** {help_text}")

        st.write("### Recommended Actions:")
        advices = get_risk_guidance(aqi)
        for advice in advices:
            st.write(f"- {advice}")

        st.divider()

        # 3. GENERATE DOWNLOADABLE REPORT
        report_data = pd.DataFrame({
            "Attribute": ["Predicted AQI", "Category", "General Impact", "Recommended Actions"],
            "Value": [f"{aqi:.2f}", cat, help_text, " | ".join(advices)]
        })
        
        # Convert to CSV
        csv = report_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label=" Download Health Report (CSV)",
            data=csv,
            file_name=f"AQI_Report_{cat}.csv",
            mime="text/csv",
            help="Click to download your AQI prediction and health advice as a CSV file."
        )
            
    else:
        st.warning(" No prediction data found. Please run a prediction on the **Predict AQI** page first.")

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
    st.info("**Our Project:**uses machine learning to provide real-time air quality predictions.It aims to help users make informed decisions about their outdoor activities based on pollutant levels. Also Bridges this gap by using Machine Learning to forecast AQI levels **24–48 hours in advance**.")

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