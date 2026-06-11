import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Visualizaciones", page_icon="📊", layout="wide")

st.markdown("# 📊 Visualizaciones Interactivas")
st.markdown(
    "Análisis visual de las principales variables relacionadas con el abandono de clientes."
)
st.markdown("---")

COLORES = {"No": "#27ae60", "Yes": "#e74c3c"}


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


def detectar_col(df, keywords, excludes=None):
    excludes = excludes or []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keywords) and not any(e in cl for e in excludes):
            return c
    return None


df, error = cargar_datos()

if error:
    st.error(f"⚠️ Error al cargar los datos: {error}")
    st.stop()

# Detectar columnas clave de forma flexible
churn_col = detectar_col(df, ["churn"], ["value", "score", "reason", "category"])
contract_col = detectar_col(df, ["contract"])
payment_col = detectar_col(df, ["payment"])
tenure_col = detectar_col(df, ["tenure"])
monthly_col = detectar_col(df, ["monthly"])

if not churn_col:
    st.error("No se encontró la columna de Churn en el dataset.")
    st.stop()

# Normalizar la columna churn a Yes/No
df[churn_col] = df[churn_col].astype(str).str.strip()
df[churn_col] = df[churn_col].replace({"1": "Yes", "0": "No", "1.0": "Yes", "0.0": "No"})

# ── Gráfico 1: Contract vs Churn ────────────────────────────────────────────
st.subheader("📋 1. Tipo de Contrato vs Abandono")

if contract_col:
    contrato_churn = (
        df.groupby([contract_col, churn_col])
        .size()
        .reset_index(name="Cantidad")
    )
    fig1 = px.bar(
        contrato_churn,
        x=contract_col,
        y="Cantidad",
        color=churn_col,
        barmode="group",
        color_discrete_map=COLORES,
        labels={contract_col: "Tipo de Contrato", churn_col: "Abandono"},
        title="Distribución de Abandono por Tipo de Contrato",
        template="plotly_white",
    )
    fig1.update_layout(legend_title="Churn", font=dict(size=13))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    > **Interpretación:** Los clientes con contratos **mes a mes** presentan la mayor tasa de
    abandono. En contraste, los contratos de **uno o dos años** generan mayor fidelización,
    posiblemente por el compromiso a largo plazo y los beneficios asociados.
    """)
else:
    st.warning("No se encontró la columna de tipo de contrato.")

st.markdown("---")

# ── Gráfico 2: Payment Method vs Churn ─────────────────────────────────────
st.subheader("💳 2. Método de Pago vs Abandono")

if payment_col:
    pago_churn = (
        df.groupby([payment_col, churn_col])
        .size()
        .reset_index(name="Cantidad")
    )
    fig2 = px.bar(
        pago_churn,
        x=payment_col,
        y="Cantidad",
        color=churn_col,
        barmode="stack",
        color_discrete_map=COLORES,
        labels={payment_col: "Método de Pago", churn_col: "Abandono"},
        title="Distribución de Abandono por Método de Pago",
        template="plotly_white",
    )
    fig2.update_layout(
        legend_title="Churn",
        xaxis_tickangle=-20,
        font=dict(size=13),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    > **Interpretación:** Los clientes que pagan mediante **cheque electrónico** muestran
    la tasa de abandono más elevada. Los métodos de pago automáticos (tarjeta de crédito,
    débito bancario) están asociados a mayor retención, posiblemente por la comodidad
    y el menor esfuerzo de renovación.
    """)
else:
    st.warning("No se encontró la columna de método de pago.")

st.markdown("---")

# ── Gráfico 3: Tenure Months vs Churn ──────────────────────────────────────
st.subheader("📅 3. Meses de Permanencia vs Abandono")

if tenure_col:
    fig3 = px.histogram(
        df,
        x=tenure_col,
        color=churn_col,
        nbins=40,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map=COLORES,
        labels={tenure_col: "Meses de Permanencia", churn_col: "Abandono"},
        title="Distribución de Meses de Permanencia por Abandono",
        template="plotly_white",
    )
    fig3.update_layout(legend_title="Churn", font=dict(size=13))
    st.plotly_chart(fig3, use_container_width=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        avg_churn = df[df[churn_col] == "Yes"][tenure_col].mean()
        st.metric("Promedio meses (Abandonaron)", f"{avg_churn:.1f}")
    with col_b2:
        avg_active = df[df[churn_col] == "No"][tenure_col].mean()
        st.metric("Promedio meses (Activos)", f"{avg_active:.1f}")

    st.markdown("""
    > **Interpretación:** La mayoría de los abandonos ocurren en los **primeros meses** de
    relación con la empresa. Clientes con mayor antigüedad tienden a permanecer activos,
    lo que sugiere que los esfuerzos de retención deben enfocarse en los primeros 12 meses.
    """)
else:
    st.warning("No se encontró la columna de meses de permanencia.")

st.markdown("---")

# ── Gráfico 4: Monthly Charges vs Churn ────────────────────────────────────
st.subheader("💰 4. Cargos Mensuales vs Abandono")

if monthly_col:
    fig4 = px.box(
        df,
        x=churn_col,
        y=monthly_col,
        color=churn_col,
        color_discrete_map=COLORES,
        points="outliers",
        labels={churn_col: "Abandono", monthly_col: "Cargos Mensuales ($)"},
        title="Distribución de Cargos Mensuales según Abandono",
        template="plotly_white",
    )
    fig4.update_layout(showlegend=False, font=dict(size=13))
    st.plotly_chart(fig4, use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        med_churn = df[df[churn_col] == "Yes"][monthly_col].median()
        st.metric("Mediana cargos (Abandonaron)", f"${med_churn:.2f}")
    with col_c2:
        med_active = df[df[churn_col] == "No"][monthly_col].median()
        st.metric("Mediana cargos (Activos)", f"${med_active:.2f}")

    st.markdown("""
    > **Interpretación:** Los clientes que abandonaron tienden a tener **cargos mensuales más
    altos** en comparación con los que permanecen activos. Una tarifa elevada combinada con
    un contrato flexible puede ser un indicador de riesgo de abandono.
    """)
else:
    st.warning("No se encontró la columna de cargos mensuales.")

st.markdown("---")

# ── Gráfico 5 (bonus): Distribución general del churn ──────────────────────
st.subheader("🥧 5. Distribución General del Abandono")

churn_counts = df[churn_col].value_counts().reset_index()
churn_counts.columns = ["Churn", "Cantidad"]

fig5 = px.pie(
    churn_counts,
    names="Churn",
    values="Cantidad",
    color="Churn",
    color_discrete_map=COLORES,
    title="Proporción de Clientes: Abandono vs Activos",
    hole=0.4,
    template="plotly_white",
)
fig5.update_traces(textposition="inside", textinfo="percent+label")
fig5.update_layout(font=dict(size=14))

st.plotly_chart(fig5, use_container_width=True)
st.markdown("""
> **Interpretación:** El dataset presenta un **desbalance de clases** típico en problemas de
churn, donde la mayoría de los clientes son activos. Este desbalance fue considerado durante
el entrenamiento del modelo para evitar sesgos en la predicción.
""")
