import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS personalizados ──────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-divider {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 24px 0;
    }
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #2d6a9f;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar_datos():
    ruta = os.path.join("data", "Telco_customer_churn.xlsx")
    try:
        df = pd.read_excel(ruta)
        return df, None
    except FileNotFoundError:
        return None, f"Archivo no encontrado: {ruta}"
    except Exception as e:
        return None, str(e)


# ── Encabezado ──────────────────────────────────────────────────────────────
st.markdown("# 📊 Sistema Inteligente de Predicción de Abandono de Clientes")
st.markdown("#### Customer Churn Analysis · Proyecto Final Integrador")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Información académica ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <strong>🎓 Institución:</strong> Universidad Casa Grande<br>
        <strong>📚 Programa:</strong> Maestría en Inteligencia Artificial y Ciencia de Datos<br>
        <strong>👩‍💻 Autora:</strong> Oreana Almeida Cedeño
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objetivo:</strong><br>
        Desarrollar un modelo predictivo que identifique clientes con alta probabilidad
        de abandono (churn), permitiendo a la empresa implementar estrategias de
        retención proactivas basadas en datos.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Métricas del dataset ────────────────────────────────────────────────────
st.subheader("📈 Resumen del Dataset")

df, error = cargar_datos()

if error:
    st.error(f"⚠️ Error al cargar los datos: {error}")
    st.info("Verifica que el archivo `data/Telco_customer_churn.xlsx` existe.")
else:
    # Detectar columna de churn (flexible ante variaciones de nombre)
    churn_col = None
    for c in df.columns:
        if (
            "churn" in c.lower()
            and "value" not in c.lower()
            and "score" not in c.lower()
            and "reason" not in c.lower()
        ):
            churn_col = c
            break

    total = len(df)
    if churn_col:
        churn_vals = df[churn_col].astype(str).str.strip().str.lower()
        abandonaron = int((churn_vals == "yes").sum() + (churn_vals == "1").sum())
        activos = total - abandonaron
        tasa = abandonaron / total * 100
    else:
        abandonaron = activos = 0
        tasa = 0.0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Clientes</div>
            <div class="metric-value">{total:,}</div>
        </div>""", unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #1a6b3a, #27ae60);">
            <div class="metric-label">Clientes Activos</div>
            <div class="metric-value">{activos:,}</div>
        </div>""", unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #7b1a1a, #c0392b);">
            <div class="metric-label">Clientes que Abandonaron</div>
            <div class="metric-value">{abandonaron:,}</div>
        </div>""", unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #6b4a0a, #e67e22);">
            <div class="metric-label">Tasa de Abandono</div>
            <div class="metric-value">{tasa:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Información del dataset ─────────────────────────────────────────────
    st.subheader("🗂️ Acerca del Dataset")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Filas", f"{df.shape[0]:,}")
    with col_d2:
        st.metric("Columnas", df.shape[1])
    with col_d3:
        st.metric("Variables numéricas", len(df.select_dtypes(include="number").columns))

    st.markdown("""
    <div class="info-box">
        <strong>📋 Descripción:</strong> El dataset <em>Telco Customer Churn</em> contiene
        información demográfica, de servicios contratados y de facturación de clientes de
        una empresa de telecomunicaciones. Ampliamente utilizado en Ciencia de Datos para
        el análisis de retención de clientes.<br><br>
        <strong>Fuente:</strong> IBM Sample Data · Telco Customer Churn Dataset
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Navegación ──────────────────────────────────────────────────────────────
st.subheader("🧭 Navegación")
st.info("Utiliza el **menú lateral izquierdo** para explorar las secciones de la aplicación.")

nav_cols = st.columns(4)
pages_info = [
    ("🔍", "Exploración", "Análisis inicial del dataset: shape, tipos, nulos y estadísticas."),
    ("📊", "Visualizaciones", "Gráficos interactivos sobre las principales variables del churn."),
    ("🤖", "Modelo", "Métricas de rendimiento del modelo de Machine Learning."),
    ("🎯", "Predicción", "Formulario para predecir el abandono de un cliente específico."),
]
for col, (icon, title, desc) in zip(nav_cols, pages_info):
    with col:
        st.markdown(f"""
        <div style="background:#f8f9fa; border-radius:10px; padding:16px; text-align:center;
                    border:1px solid #dee2e6; height:140px;">
            <div style="font-size:2rem;">{icon}</div>
            <strong>{title}</strong>
            <p style="font-size:0.8rem; color:#555; margin-top:6px;">{desc}</p>
        </div>""", unsafe_allow_html=True)
