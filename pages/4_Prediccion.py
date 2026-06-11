import streamlit as st
import pandas as pd
import pickle
st.set_page_config(page_title="Predicción de Churn", page_icon="🎯", layout="wide")

st.markdown("# 🎯 Predicción de Abandono de Cliente")
st.markdown(
    "Completa el formulario con los datos del cliente para predecir su probabilidad "
    "de abandono utilizando el modelo entrenado."
)
st.markdown("---")


# ── Carga del modelo y columnas ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    try:
        with open("modelo_churn.pkl", "rb") as f:
            modelo = pickle.load(f)
        with open("columnas_modelo.pkl", "rb") as f:
            columnas = pickle.load(f)
        return modelo, columnas, None
    except FileNotFoundError as e:
        return None, None, f"Archivo no encontrado: {e}"
    except Exception as e:
        return None, None, str(e)


modelo, columnas, error_modelo = cargar_modelo()

if error_modelo:
    st.error(f"⚠️ Error al cargar el modelo: {error_modelo}")
    st.stop()

st.success("✅ Modelo cargado correctamente.")

# Mostrar columnas esperadas por el modelo (expandible)
with st.expander("ℹ️ Ver columnas utilizadas por el modelo"):
    if isinstance(columnas, (list, tuple)):
        st.write(columnas)
    else:
        st.write(columnas)

st.markdown("---")

# ── Formulario de entrada ───────────────────────────────────────────────────
st.subheader("📝 Datos del Cliente")

with st.form("formulario_prediccion"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Información Demográfica**")
        gender = st.selectbox("Género", ["Male", "Female"])
        senior_citizen = st.selectbox("¿Es ciudadano mayor?", [0, 1],
                                       format_func=lambda x: "Sí" if x == 1 else "No")
        partner = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])

    with col2:
        st.markdown("**Servicios Contratados**")
        tenure_months = st.slider("Meses como cliente", 0, 72, 12)
        phone_service = st.selectbox("Servicio telefónico", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Múltiples líneas", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Servicio de internet", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Seguridad en línea", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Respaldo en línea", ["No", "Yes", "No internet service"]
        )

    with col3:
        st.markdown("**Contrato y Facturación**")
        device_protection = st.selectbox(
            "Protección de dispositivo", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Soporte técnico", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming películas", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Tipo de contrato", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Factura sin papel", ["Yes", "No"])
        payment_method = st.selectbox(
            "Método de pago",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Cargos mensuales ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
        )
        total_charges = st.number_input(
            "Cargos totales ($)", min_value=0.0, max_value=10000.0,
            value=float(tenure_months * 65), step=1.0
        )

    st.markdown("---")
    submitted = st.form_submit_button(
        "🔮 Predecir Abandono", use_container_width=True, type="primary"
    )


# ── Lógica de predicción ────────────────────────────────────────────────────
if submitted:
    # Construir diccionario con todos los campos del formulario
    datos_raw = {
        "gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure_months,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
    }

    try:
        # Crear DataFrame con una sola fila
        df_input = pd.DataFrame([datos_raw])

        # Obtener lista de columnas esperadas
        if isinstance(columnas, (list, tuple)):
            cols_esperadas = list(columnas)
        elif hasattr(columnas, "tolist"):
            cols_esperadas = columnas.tolist()
        else:
            cols_esperadas = list(columnas)

        # Aplicar get_dummies para codificación one-hot
        df_encoded = pd.get_dummies(df_input)

        # Alinear con las columnas del modelo (agregar columnas faltantes con 0)
        df_final = df_encoded.reindex(columns=cols_esperadas, fill_value=0)

        # Ejecutar predicción
        prediccion = modelo.predict(df_final)[0]
        probabilidades = modelo.predict_proba(df_final)[0]

        # La probabilidad de churn (clase 1 = Yes)
        clases = list(modelo.classes_)
        idx_churn = clases.index(1) if 1 in clases else clases.index("Yes") if "Yes" in clases else 1
        prob_churn = probabilidades[idx_churn] * 100

        st.markdown("---")
        st.subheader("📊 Resultado de la Predicción")

        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            st.metric("Probabilidad de Abandono", f"{prob_churn:.1f}%")

            # Barra de progreso visual
            color_barra = (
                "#e74c3c" if prob_churn > 70 else "#e67e22" if prob_churn > 40 else "#27ae60"
            )
            st.markdown(f"""
            <div style="background:#e0e0e0; border-radius:8px; height:20px; margin:8px 0;">
                <div style="background:{color_barra}; width:{prob_churn:.1f}%; height:100%;
                            border-radius:8px; transition: width 0.5s ease;"></div>
            </div>
            <p style="text-align:center; color:{color_barra}; font-weight:600;">
                {prob_churn:.1f}% de probabilidad de churn
            </p>
            """, unsafe_allow_html=True)

        with col_r2:
            if prob_churn < 40:
                st.success(f"""
                ✅ **Riesgo BAJO de abandono** ({prob_churn:.1f}%)

                El cliente muestra un perfil estable. No se requieren acciones inmediatas.
                Se recomienda mantener la calidad del servicio y contacto periódico.
                """)
            elif prob_churn <= 70:
                st.warning(f"""
                ⚠️ **Riesgo MEDIO de abandono** ({prob_churn:.1f}%)

                El cliente presenta señales moderadas de posible abandono.
                Se recomienda contacto proactivo, oferta de beneficios o revisión del plan contratado.
                """)
            else:
                st.error(f"""
                🚨 **Riesgo ALTO de abandono** ({prob_churn:.1f}%)

                El cliente tiene alta probabilidad de abandonar el servicio.
                Se recomienda intervención inmediata: oferta de descuento, mejora del plan o
                llamada de retención personalizada.
                """)

        # Tabla de probabilidades por clase
        st.markdown("**Distribución de probabilidades:**")
        prob_df = pd.DataFrame({
            "Clase": ["No Abandona", "Abandona"],
            "Probabilidad": [f"{probabilidades[1 - idx_churn]*100:.1f}%", f"{prob_churn:.1f}%"],
        })
        st.table(prob_df)

        # Resumen de datos ingresados
        with st.expander("🔍 Ver datos ingresados al modelo"):
            st.dataframe(df_final, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error durante la predicción: {e}")
        st.info(
            "Verifica que los datos ingresados sean compatibles con el modelo entrenado. "
            "Revisa las columnas esperadas en el expansor superior."
        )
