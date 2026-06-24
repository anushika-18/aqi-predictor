import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import date

# ── Load model (predicts AQI_Bucket 0-5, NOT raw AQI score) ──
model   = joblib.load('model/aqi_model.pkl')

# AQI_Bucket mapping from your dataset
AQI_LABELS = {
    0: ("Good",         "#2ecc71", "😊"),
    1: ("Satisfactory", "#27ae60", "🙂"),
    2: ("Moderate",     "#f39c12", "😐"),
    3: ("Poor",         "#e67e22", "😷"),
    4: ("Very Poor",    "#e74c3c", "🤧"),
    5: ("Severe",       "#8e44ad", "☠️"),
}

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }
    .main { background-color: #0d1117; }

    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 0.6s ease;
    }
    .result-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    .result-emoji { font-size: 3.5rem; }
    .result-desc  { font-size: 1rem; opacity: 0.85; margin-top: 0.4rem; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stNumberInput input, .stSelectbox select {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stDateInput input {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
    }

    .legend-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .legend-chip {
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🌫️ Air Quality Index Predictor")
st.markdown("Enter pollutant levels and location to predict the air quality category.")

st.markdown("""
<div class="legend-row">
  <span class="legend-chip" style="background:#2ecc71">😊 Good</span>
  <span class="legend-chip" style="background:#27ae60">🙂 Satisfactory</span>
  <span class="legend-chip" style="background:#f39c12">😐 Moderate</span>
  <span class="legend-chip" style="background:#e67e22">😷 Poor</span>
  <span class="legend-chip" style="background:#e74c3c">🤧 Very Poor</span>
  <span class="legend-chip" style="background:#8e44ad">☠️ Severe</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── City & Date ───────────────────────────────────────────────
city_map = {
    "Delhi": 0, "Mumbai": 1, "Chennai": 2,
    "Kolkata": 3, "Bangalore": 4, "Hyderabad": 5
}

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("🏙️ City", list(city_map.keys()))
with col2:
    selected_date = st.date_input("📅 Date", value=date.today())
    year        = selected_date.year
month       = selected_date.month
day         = selected_date.day
day_of_week = selected_date.weekday()

# ── Pollutant Inputs ──────────────────────────────────────────
st.markdown("### 🧪 Pollutant Levels")
col3, col4, col5 = st.columns(3)

with col3:
    pm25    = st.number_input("PM2.5  (µg/m³)",  min_value=0.0, value=45.0, step=0.5)
    pm10    = st.number_input("PM10   (µg/m³)",  min_value=0.0, value=80.0, step=0.5)
    no      = st.number_input("NO     (µg/m³)",  min_value=0.0, value=10.0, step=0.1)
    no2     = st.number_input("NO2    (µg/m³)",  min_value=0.0, value=25.0, step=0.1)
with col4:
    nox     = st.number_input("NOx    (µg/m³)",  min_value=0.0, value=35.0, step=0.1)
    nh3     = st.number_input("NH3    (µg/m³)",  min_value=0.0, value=15.0, step=0.1)
    co      = st.number_input("CO     (mg/m³)",  min_value=0.0, value=1.5,  step=0.1)
    so2     = st.number_input("SO2    (µg/m³)",  min_value=0.0, value=12.0, step=0.1)
with col5:
    o3      = st.number_input("O3     (µg/m³)",  min_value=0.0, value=30.0, step=0.5)
    benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, value=2.0,  step=0.1)
    toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, value=5.0,  step=0.1)
    xylene  = st.number_input("Xylene  (µg/m³)", min_value=0.0, value=1.0,  step=0.1)

st.markdown("")

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍 Predict Air Quality", use_container_width=True):
    input_data = pd.DataFrame([[
        city_map[city], pm25, pm10, no, no2, nox,
        nh3, co, so2, o3, benzene, toluene, xylene,
        year, month, day, day_of_week
    ]], columns=model.feature_names_in_)

    # Model predicts AQI_Bucket (0–5)
    bucket = int(model.predict(input_data)[0])
    bucket = max(0, min(5, bucket))   # clamp just in case

    label, color, emoji = AQI_LABELS[bucket]

    # ── Result card ───────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card" style="background:{color}22; border: 2px solid {color};">
        <div class="result-emoji">{emoji}</div>
        <p class="result-label" style="color:{color}">{label}</p>
        <p class="result-desc">AQI Bucket: {bucket} &nbsp;|&nbsp; City: {city} &nbsp;|&nbsp; Date: {selected_date}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────
    st.markdown("#### 📊 Input Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PM2.5", f"{pm25} µg/m³")
    m2.metric("PM10",  f"{pm10} µg/m³")
    m3.metric("NO2",   f"{no2} µg/m³")
    m4.metric("CO",    f"{co} mg/m³")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("SO2",     f"{so2} µg/m³")
    m6.metric("O3",      f"{o3} µg/m³")
    m7.metric("Benzene", f"{benzene} µg/m³")
    m8.metric("NOx",     f"{nox} µg/m³")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;opacity:0.4;font-size:0.8rem;'>AQI Predictor • India Air Quality • Data from CPCB</p>",
    unsafe_allow_html=True
)