# ============================================================
# 🚨 CRIME HOTSPOT PREDICTION SYSTEM
# Developed by: Sanjana Mali
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Crime Hotspot Prediction",
    page_icon="🚨",
    layout="wide"
)

# ------------------------------------------------------------
# LIGHT THEME UI
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #f8fbff, #eef3f9);
    color: #1f2937;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3 {
    color: #0f172a;
}

.brand-box {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.logo img {
    filter: brightness(1.2);
}

.stButton>button {
    background: linear-gradient(45deg, #2563eb, #1e40af);
    color: white;
    border-radius: 25px;
    padding: 10px 28px;
    font-size: 16px;
}

.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD DATA & MODELS
# ------------------------------------------------------------
df = pd.read_csv("crime_data_updated.csv")

# FIX: Ensure Total_Crime is numeric
df["Total_Crime"] = pd.to_numeric(df["Total_Crime"], errors="coerce")
df = df.dropna(subset=["Total_Crime"])

model = pickle.load(open("finalized_RFmodel.sav", "rb"))
scaler = pickle.load(open("scaler_model.sav", "rb"))

# ------------------------------------------------------------
# PROGRAM BRANDING
# ------------------------------------------------------------
st.markdown("<div class='brand-box'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.image("assets/edunet.png", width=140)
with col2:
    st.image("assets/microsoft.png", width=140)
with col3:
    st.image("assets/sap.png", width=140)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# CITY COORDINATES
# ------------------------------------------------------------
city_coordinates = {
    "Ahmedabad": [23.0225, 72.5714],
    "Bengaluru": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Delhi": [28.7041, 77.1025],
    "Hyderabad": [17.3850, 78.4867],
    "Kolkata": [22.5726, 88.3639],
    "Mumbai": [19.0760, 72.8777],
    "Pune": [18.5204, 73.8567],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462]
}

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("🚓 Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🌍 Crime Map", "🤖 Prediction"]
)

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
if menu == "🏠 Overview":
    st.title("🚨 Crime Hotspot Prediction System")
    st.subheader("AI-powered decision support for safer cities")

    st.markdown("""
    ### 🔍 What this app does
    - Predicts **crime risk levels** for Indian metro cities  
    - Identifies **high-risk hotspots** using machine learning  
    - Supports **data-driven policing & urban safety planning**

    ### How it works
    - Trained ML models analyze historical crime patterns  
    - City, year & population are used for prediction  
    - Outputs **Low / Medium / High crime risk**

    ### Impact
    - Helps authorities plan patrols efficiently  
    - Encourages proactive crime prevention  
    """)

# ------------------------------------------------------------
# CRIME MAP
# ------------------------------------------------------------

elif menu == "🌍 Crime Map":
    st.title("🌍 Crime Hotspot Map of India")

    crime_city = df.groupby("City")["Total_Crime"].sum().reset_index()

    # 🔹 Dynamic thresholds (relative comparison)
    high_threshold = crime_city["Total_Crime"].quantile(0.75)
    medium_threshold = crime_city["Total_Crime"].quantile(0.40)

    india_map = folium.Map(location=[22.5, 78.9], zoom_start=5)

    for _, row in crime_city.iterrows():
        city = row["City"]
        crime = row["Total_Crime"]

        if city in city_coordinates:
            lat, lon = city_coordinates[city]

            if crime >= high_threshold:
                color = "red"
            elif crime >= medium_threshold:
                color = "orange"
            else:
                color = "green"

            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=f"<b>{city}</b><br>Total Crime: {int(crime)}",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(india_map)

    st_folium(india_map, width=900, height=520)

    st.markdown("""
### 📊 2025 Crime Map Analysis
- 🔴 **High Risk**: Cities in the top 25% crime range
- 🟠 **Medium Risk**: Cities with moderate crime levels
- 🟢 **Low Risk**: Cities with comparatively lower crime
  
⚠️ Risk levels are **relative**, not absolute.
""")


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
elif menu == "🤖 Prediction":
    st.title("🤖 Crime Risk Prediction")

    col1, col2, col3 = st.columns(3)

    year = col1.slider("Year", 2022, 2030, 2025)
    city = col2.selectbox("City", sorted(df["City"].dropna().unique()))
    population = col3.number_input("Population (in Lakhs)", value=100.0)

    if st.button("🚀 Predict Crime Risk"):
        features = model.feature_names_in_
        sample = pd.DataFrame(np.zeros((1, len(features))), columns=features)

        sample["Year"] = year
        sample["Population (in Lakhs) (2011)+"] = population

        city_col = f"City_{city}"
        if city_col in sample.columns:
            sample[city_col] = 1

        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]

        if prediction == "High":
            st.error("🚨 HIGH CRIME RISK ZONE")
        elif prediction == "Medium":
            st.warning("⚠️ MODERATE CRIME RISK")
        else:
            st.success("✅ LOW CRIME RISK")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("👩‍💻 **Developed by Sanjana Mali | AI & Data Science**")
