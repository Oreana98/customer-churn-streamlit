import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Exploración de Datos", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Exploración del Dataset")
st.markdown("Análisis exploratorio inicial del dataset *Telco Customer Churn*.")

st.markdown("---")


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


df, error = cargar_datos()

if error:
    st.error(f"⚠️ Error al cargar los datos: {error}")
    st.stop()

# ── 1. Dimensiones ──────────────────────────────────────────────────────────
st.subheader("📐 Dimensiones del Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Filas", f"{df.shape[0]:,}")
col2.metric("Columnas", df.shape[1])
col3.metric("Total celdas", f"{df.shape[0] * df.shape[1]:,}")

st.caption(f"Shape: `{df.shape}`")

st.markdown("---")

# ── 2. Primeras filas ───────────────────────────────────────────────────────
st.subheader("👀 Primeras Filas del Dataset")

n_rows = st.slider("Número de filas a mostrar", min_value=3, max_value=20, value=5)
st.dataframe(df.head(n_rows), use_container_width=True)

st.markdown("---")

# ── 3. Tipos de datos ───────────────────────────────────────────────────────
st.subheader("🔤 Tipos de Datos por Columna")

tipos = df.dtypes.reset_index()
tipos.columns = ["Columna", "Tipo de Dato"]
tipos["Tipo de Dato"] = tipos["Tipo de Dato"].astype(str)

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.dataframe(tipos, use_container_width=True, height=350)

with col_t2:
    resumen_tipos = tipos["Tipo de Dato"].value_counts().reset_index()
    resumen_tipos.columns = ["Tipo", "Cantidad"]

    st.markdown("**Resumen por tipo:**")
    for _, row in resumen_tipos.iterrows():
        st.markdown(f"- `{row['Tipo']}`: **{row['Cantidad']}** columnas")

    num_cols = len(df.select_dtypes(include="number").columns)
    cat_cols = len(df.select_dtypes(include="object").columns)
    st.info(f"🔢 Numéricas: {num_cols} &nbsp;&nbsp;|&nbsp;&nbsp; 🔡 Categóricas: {cat_cols}")

st.markdown("---")

# ── 4. Valores nulos ────────────────────────────────────────────────────────
st.subheader("❓ Valores Nulos")

nulos = df.isnull().sum().reset_index()
nulos.columns = ["Columna", "Valores Nulos"]
nulos["Porcentaje (%)"] = (nulos["Valores Nulos"] / len(df) * 100).round(2)
nulos = nulos[nulos["Valores Nulos"] > 0].sort_values("Valores Nulos", ascending=False)

if nulos.empty:
    st.success("✅ El dataset no contiene valores nulos.")
else:
    st.warning(f"⚠️ Se encontraron valores nulos en {len(nulos)} columna(s).")
    st.dataframe(
        nulos.style.background_gradient(subset=["Porcentaje (%)"], cmap="Reds"),
        use_container_width=True,
    )

st.markdown("---")

# ── 5. Estadísticas descriptivas ────────────────────────────────────────────
st.subheader("📊 Estadísticas Descriptivas")

tab_num, tab_cat = st.tabs(["Variables Numéricas", "Variables Categóricas"])

with tab_num:
    desc_num = df.describe().T.round(2)
    st.dataframe(desc_num, use_container_width=True)
    st.caption(
        "count: observaciones válidas · mean: media · std: desviación estándar · "
        "min/max: valores extremos · 25%/50%/75%: percentiles."
    )

with tab_cat:
    desc_cat = df.select_dtypes(include="object").describe().T
    if desc_cat.empty:
        st.info("No hay variables categóricas en el dataset.")
    else:
        st.dataframe(desc_cat, use_container_width=True)
        st.caption(
            "count: observaciones · unique: valores distintos · "
            "top: valor más frecuente · freq: frecuencia del valor top."
        )

st.markdown("---")

# ── 6. Vista completa filtrada ──────────────────────────────────────────────
with st.expander("🗃️ Explorar Dataset Completo"):
    columnas_sel = st.multiselect(
        "Selecciona columnas a visualizar:",
        options=df.columns.tolist(),
        default=df.columns[:8].tolist(),
    )
    if columnas_sel:
        st.dataframe(df[columnas_sel], use_container_width=True, height=400)
    else:
        st.warning("Selecciona al menos una columna.")
