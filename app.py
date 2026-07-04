import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Riesgo Actuarial IA",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS NUEVO (AZUL + FORMULARIO DESTACADO)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #e3f2fd, #f5f9ff);
}

/* TÍTULO */
.main-title {
    text-align: center;
    color: #0d47a1;
    font-size: 2.4rem;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #546e7a;
    margin-bottom: 2rem;
}

/* TARJETA DEL FORMULARIO */
.form-card {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border: 1px solid #bbdefb;
}

/* RESULTADO */
.result-card {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
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
    background: #1976d2;
    color: white;
    font-weight: bold;
    font-size: 1.2rem;
}

/* BOTÓN */
.stButton>button {
    background: #1976d2;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background: #0d47a1;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 class='main-title'>📊 Clasificador de Riesgo Actuarial</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>IA-ISC • KMeans + Pipeline • 2026</p>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#607d8b; margin-bottom:2rem;">
Sistema inteligente de análisis de riesgo de clientes aseguradores
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODELOS
# ============================================================
MODEL_DIR = Path("models")

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_riesgo_actuarial.pkl"

@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    return preprocessor, kmeans

preprocessor, modelo = cargar_modelos()

# ============================================================
# FORMULARIO (DESTACADO EN TARJETA)
# ============================================================
st.markdown("<div class='form-card'>", unsafe_allow_html=True)

st.markdown("### 🧾 Datos del cliente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", 18, 100, 30)

    sex = st.radio(
        "Sexo",
        ["male", "female"],
        horizontal=True
    )

    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

with col2:
    children = st.number_input("Hijos", 0, 10, 0)

    smoker = st.radio(
        "Fumador",
        ["yes", "no"],
        horizontal=True
    )

    region = st.selectbox(
        "Región",
        ["southeast", "southwest", "northeast", "northwest"]
    )

charges = st.number_input("Gastos médicos", 0, 100000, 5000)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICCIÓN
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

    X_transformed = preprocessor.transform(cliente)
    cluster = modelo.predict(X_transformed)[0]

    interpretacion = {
        0: "Riesgo Bajo 🟢",
        1: "Riesgo Medio 🟡",
        2: "Riesgo Alto 🔴",
        3: "Riesgo Crítico ⚠️"
    }

    resultado = interpretacion.get(cluster, f"Cluster {cluster}")

    st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3rem;">📊</div>
            <div class="cluster-badge">{resultado}</div>
            <p style="margin-top:1rem;">Cluster asignado: {cluster}</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("Ingrese los datos del cliente y presione Predecir riesgo")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; margin-top:3rem; color:#90a4ae;">
📊 Sistema de Riesgo Actuarial con IA • ISC 2026
</div>
""", unsafe_allow_html=True)
