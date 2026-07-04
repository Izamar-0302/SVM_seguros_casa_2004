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
# CSS (DEL DISEÑO BONITO DE FLORES)
# ============================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .main-title {
            text-align: center;
            color: #2E7D32;
            font-weight: 700;
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
        }
        
        .subtitle {
            text-align: center;
            color: #558B2F;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .result-card {
            background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
            border-radius: 20px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 8px 32px rgba(46, 125, 50, 0.15);
            border: 1px solid #A5D6A7;
            text-align: center;
        }
        
        .winner-badge {
            display: inline-block;
            background: linear-gradient(135deg, #2E7D32, #43A047);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.3rem;
            box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER (FLORES STYLE)
# ============================================================
st.markdown('<div style="font-size:3rem; text-align:center;">📊</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Clasificador de Riesgo Actuarial</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">IA-ISC • KMeans + Pipeline • 2026</p>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #616161; margin-bottom: 2rem;">
        🧠 Sistema inteligente de análisis de riesgo de clientes
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
# INPUTS
# ============================================================
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

    region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])

charges = st.number_input("Gastos médicos", 0, 100000, 5000)

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
            <div class="winner-badge">{resultado}</div>
            <p style="margin-top:1rem;">Cluster asignado: {cluster}</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("Ingrese los datos del cliente y presione Predecir riesgo")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div style="text-align:center; margin-top:3rem; color:#9E9E9E;">
        📊 Sistema de Riesgo Actuarial con IA • ISC 2026
    </div>
""", unsafe_allow_html=True)
