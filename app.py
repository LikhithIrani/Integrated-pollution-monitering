import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt


# ---------------- LOAD FILES ----------------

water_model = joblib.load("water_model.pkl")
air_model = joblib.load("air_model.pkl")
soil_model = joblib.load("soil_model.pkl")

water_scaler = joblib.load("water_scaler.pkl")
air_scaler = joblib.load("air_scaler.pkl")
soil_scaler = joblib.load("soil_scaler.pkl")

water_cols = joblib.load("water_features.pkl")
air_cols = joblib.load("air_features.pkl")
soil_cols = joblib.load("soil_features.pkl")


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Integrated Pollution Monitor",
    layout="centered"
)

st.title("🌍 Integrated Pollution Monitoring System")


# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs(["💧 Water", "🌫️ Air", "🌱 Soil"])


# ================= WATER =================

with tab1:

    st.header("💧 Water Quality Analysis")

    w_vals = []

    for c in water_cols:
        v = st.number_input(
            label=c,
            key="w"+c,
            value=0.0,
            format="%.4f"
        )
        w_vals.append(v)

    col1, col2 = st.columns(2)

    with col1:
        analyze_water = st.button("Analyze Water")

    with col2:
        graph_water = st.button("Show Water Graph")

    if analyze_water:

        arr = np.array(w_vals).reshape(1, -1)
        arr = water_scaler.transform(arr)

        pred = water_model.predict(arr)[0]
        prob = water_model.predict_proba(arr)[0][1]

        score = round(prob * 100, 2)

        if pred == 1:
            st.success("✅ Water is SAFE")
        else:
            st.error("❌ Water is UNSAFE")

        st.metric("Health Score", f"{score} %")


    if graph_water:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(water_cols, w_vals)

        ax.set_title("Water Quality Parameters")
        ax.set_ylabel("Concentration")
        ax.set_xlabel("Parameters")

        plt.xticks(rotation=90)

        st.pyplot(fig)



# ================= AIR =================

with tab2:

    st.header("🌫️ Air Quality Analysis")

    a_vals = []

    for c in air_cols:
        v = st.number_input(
            label=c,
            key="a"+c,
            value=0.0,
            format="%.4f"
        )
        a_vals.append(v)


    col1, col2 = st.columns(2)

    with col1:
        analyze_air = st.button("Analyze Air")

    with col2:
        graph_air = st.button("Show Air Graph")


    if analyze_air:

        arr = np.array(a_vals).reshape(1, -1)
        arr = air_scaler.transform(arr)

        pred = air_model.predict(arr)[0]

        if pred == 1:
            st.success("✅ Air is CLEAN")
        else:
            st.error("❌ Air is POLLUTED")


    if graph_air:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(air_cols, a_vals, marker="o")

        ax.set_title("Air Pollution Parameters")
        ax.set_ylabel("Level")
        ax.set_xlabel("Parameters")

        plt.xticks(rotation=90)

        st.pyplot(fig)



# ================= SOIL =================

with tab3:

    st.header("🌱 Soil Quality Analysis")

    s_vals = []

    for c in soil_cols:
        v = st.number_input(
            label=c,
            key="s"+c,
            value=0.0,
            format="%.4f"
        )
        s_vals.append(v)


    col1, col2 = st.columns(2)

    with col1:
        analyze_soil = st.button("Analyze Soil")

    with col2:
        graph_soil = st.button("Show Soil Graph")


    if analyze_soil:

        arr = np.array(s_vals).reshape(1, -1)
        arr = soil_scaler.transform(arr)

        pred = soil_model.predict(arr)[0]

        if pred == 1:
            st.success("✅ Soil is HEALTHY")
        else:
            st.error("❌ Soil is CONTAMINATED")


    if graph_soil:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(soil_cols, s_vals)

        ax.set_title("Soil Pollution Parameters")
        ax.set_xlabel("Level")

        st.pyplot(fig)


# ---------------- FOOTER ----------------

st.markdown("---")
st.markdown("📊 **Integrated Pollution Monitoring System** | ML-Based Environmental Analysis")
