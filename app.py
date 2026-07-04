import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Riesgo Actuarial IA",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        .main-title {
            text-align: center;
            color: #1E88E5;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .subtitle {
            text-align: center;
            color: #546E7A;
            margin-bottom: 2rem;
        }

        .result-card {
            background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin-top: 2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }

        .cluster-badge {
            display: inline-block;
            padding: 0.6rem 1.5rem;
            border-radius: 50px;
            background: #1976D2;
            color: white;
            font-weight: bold;
            font-size: 1.2rem;
            margin-top: 1rem;
        }

        .footer {
            text-align: center;
            color: #9E9E9E;
            margin-top: 3rem;
            font-size: 0.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 class='main-title'>📊 Clasificador de Riesgo Actuarial</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>IA-ISC • KMeans + Pipeline • 2026</p>", unsafe_allow_html=True)

# ============================================================
# MODELO
# ============================================================
MODEL_PATH = Path("models/kmeans_riesgo_actuarial.pkl")

@st.cache_resource
def cargar_modelo():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    st.error("❌ No se encontró el modelo en /models")
    st.stop()

modelo = cargar_modelo()

# ============================================================
# INTERFAZ DE ENTRADA
# ============================================================
st.markdown("### 🧾 Datos del cliente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", 18, 100, 30)
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

with col2:
    children = st.number_input("Hijos", 0, 10, 0)
    smoker = st.selectbox("Fumador", ["yes", "no"])
    region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])

charges = st.number_input("Gastos médicos", 0, 100000, 5000)

# ============================================================
# BOTÓN DE PREDICCIÓN
# ============================================================
if st.button("🔍 Predecir riesgo"):

    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])

    cluster = modelo.predict(cliente)[0]

    # Interpretación simple (ajústalo si tienes mapa_riesgo)
    interpretacion = {
        0: "Riesgo Bajo 🟢",
        1: "Riesgo Medio 🟡",
        2: "Riesgo Alto 🔴",
        3: "Riesgo Crítico ⚠️"
    }

    resultado = interpretacion.get(cluster, f"Cluster {cluster}")

    # ============================================================
    # RESULTADO
    # ============================================================
    st.markdown(f"""
        <div class="result-card">
            <h3>Resultado del análisis</h3>
            <div class="cluster-badge">{resultado}</div>
            <p style="margin-top:1rem;">Cluster asignado: {cluster}</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("Ingrese los datos del cliente y presione **Predecir riesgo**")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        📊 Sistema de Riesgo Actuarial con IA • Proyecto ISC 2026
    </div>
""", unsafe_allow_html=True)
