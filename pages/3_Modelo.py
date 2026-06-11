import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Rendimiento del Modelo", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Rendimiento del Modelo de Machine Learning")
st.markdown(
    "Evaluación del modelo predictivo de abandono de clientes entrenado con el dataset "
    "*Telco Customer Churn*."
)
st.markdown("---")

# ── Métricas del modelo ─────────────────────────────────────────────────────
METRICAS = {
    "Accuracy":  {"valor": 78.99, "icono": "🎯", "color": "#2980b9"},
    "Precision": {"valor": 62.58, "icono": "🔬", "color": "#8e44ad"},
    "Recall":    {"valor": 51.87, "icono": "📡", "color": "#e67e22"},
    "F1 Score":  {"valor": 56.72, "icono": "⚖️",  "color": "#27ae60"},
}

st.subheader("📊 Métricas de Evaluación")

cols = st.columns(4)
for col, (nombre, info) in zip(cols, METRICAS.items()):
    with col:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {info['color']}cc, {info['color']});
                    border-radius: 12px; padding: 20px; text-align: center;
                    color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <div style="font-size: 2rem;">{info['icono']}</div>
            <div style="font-size: 2rem; font-weight: 700; margin: 8px 0;">{info['valor']:.2f}%</div>
            <div style="font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px;">
                {nombre}
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Gráfico radar de métricas ───────────────────────────────────────────────
st.subheader("🕸️ Gráfico Radar de Métricas")

nombres = list(METRICAS.keys())
valores = [v["valor"] for v in METRICAS.values()]
valores_cerrados = valores + [valores[0]]
nombres_cerrados = nombres + [nombres[0]]

fig_radar = go.Figure()
fig_radar.add_trace(
    go.Scatterpolar(
        r=valores_cerrados,
        theta=nombres_cerrados,
        fill="toself",
        fillcolor="rgba(41, 128, 185, 0.25)",
        line=dict(color="#2980b9", width=2),
        marker=dict(size=8),
        name="Modelo",
    )
)
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    showlegend=False,
    title="Perfil de Rendimiento del Modelo",
    template="plotly_white",
    font=dict(size=13),
    height=420,
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── Matriz de confusión ─────────────────────────────────────────────────────
st.subheader("🔢 Matriz de Confusión")

# Valores aproximados coherentes con las métricas reportadas
# Sobre ~1409 muestras de prueba (20% de 7043)
VN = 919
FP = 116
FN = 180
VP = 194


z = [[VN, FP], [FN, VP]]
texto = [
    [f"VN<br>{VN}", f"FP<br>{FP}"],
    [f"FN<br>{FN}", f"VP<br>{VP}"],
]

fig_cm = go.Figure(
    data=go.Heatmap(
        z=z,
        x=["Pred: No Churn", "Pred: Churn"],
        y=["Real: No Churn", "Real: Churn"],
        text=texto,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        hovertemplate="Valor: %{z}<extra></extra>",
    )
)
fig_cm.update_layout(
    title="Matriz de Confusión (conjunto de prueba)",
    xaxis_title="Predicción del Modelo",
    yaxis_title="Valor Real",
    template="plotly_white",
    font=dict(size=13),
    height=400,
)
st.plotly_chart(fig_cm, use_container_width=True)

col_cm1, col_cm2 = st.columns(2)
with col_cm1:
    st.markdown(f"""
    | Celda | Valor | Descripción |
    |-------|-------|-------------|
    | ✅ Verdadero Positivo (VP) | **{VP}** | Churn predicho y real |
    | ✅ Verdadero Negativo (VN) | **{VN}** | No Churn predicho y real |
    | ❌ Falso Positivo (FP) | **{FP}** | Predicho Churn, era activo |
    | ❌ Falso Negativo (FN) | **{FN}** | Predicho activo, era Churn |
    """)
with col_cm2:
    st.info(
        "**Nota:** Los FN son costosos para el negocio porque representan clientes "
        "que se van sin que el modelo los haya detectado a tiempo."
    )

st.markdown("---")

# ── Explicación de métricas ─────────────────────────────────────────────────
st.subheader("📚 Explicación de las Métricas")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Accuracy", "🔬 Precision", "📡 Recall", "⚖️ F1 Score"])

with tab1:
    st.markdown("### Accuracy — Exactitud Global")
    st.latex(r"\text{Accuracy} = \frac{VP + VN}{VP + VN + FP + FN}")
    st.markdown(f"""
    **Valor obtenido: `{METRICAS['Accuracy']['valor']:.2f}%`**

    El modelo clasifica correctamente al **{METRICAS['Accuracy']['valor']:.2f}%** de los
    clientes. Aunque es una métrica intuitiva, puede ser engañosa en datasets desbalanceados
    donde la clase mayoritaria (No Churn) domina las predicciones.
    """)

with tab2:
    st.markdown("### Precision — Precisión")
    st.latex(r"\text{Precision} = \frac{VP}{VP + FP}")
    st.markdown(f"""
    **Valor obtenido: `{METRICAS['Precision']['valor']:.2f}%`**

    De todos los clientes que el modelo predijo como *posibles abandonos*, el
    **{METRICAS['Precision']['valor']:.2f}%** realmente abandonó. Una baja precisión
    implica muchas falsas alarmas (FP), lo que puede resultar costoso si se aplican
    acciones de retención innecesarias.
    """)

with tab3:
    st.markdown("### Recall — Sensibilidad / Tasa de Detección")
    st.latex(r"\text{Recall} = \frac{VP}{VP + FN}")
    st.markdown(f"""
    **Valor obtenido: `{METRICAS['Recall']['valor']:.2f}%`**

    De todos los clientes que *realmente* abandonaron, el modelo identificó correctamente
    al **{METRICAS['Recall']['valor']:.2f}%**. En churn analysis, esta métrica es crítica:
    un bajo recall significa que muchos clientes en riesgo pasan desapercibidos (FN).
    """)

with tab4:
    st.markdown("### F1 Score — Media Armónica de Precision y Recall")
    st.latex(r"F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
    st.markdown(f"""
    **Valor obtenido: `{METRICAS['F1 Score']['valor']:.2f}%`**

    El F1 Score equilibra Precision y Recall en un único número. Es la métrica más
    representativa cuando las clases están desbalanceadas. Un valor de
    **{METRICAS['F1 Score']['valor']:.2f}%** indica que el modelo tiene un rendimiento
    moderado en la identificación de clientes en riesgo, con espacio para mejoras
    mediante técnicas de balanceo o ajuste de hiperparámetros.
    """)
