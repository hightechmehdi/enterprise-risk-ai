import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("FASTAPI_URL").rstrip("/")
TIMEOUT = 15

st.set_page_config(page_title="Enterprise Risk Bankruptcy Prediction", page_icon="📉", layout="centered")
st.title("📉 Enterprise Risk Bankruptcy Prediction")
st.caption(f"API FastAPI : {API_BASE_URL}")


def call_api(method: str, path: str, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            timeout=TIMEOUT,
            **kwargs,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        detail = str(exc)
        if getattr(exc, "response", None) is not None:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except ValueError:
                detail = exc.response.text
        return None, detail


st.header("État du service")
if st.button("Vérifier /health", use_container_width=True):
    data, error = call_api("GET", "/health")
    if error:
        st.error(f"Erreur /health : {error}")
    else:
        if data.get("model_loaded"):
            st.success("API disponible et modèle chargé")
        else:
            st.warning("API disponible, mais aucun modèle n'est chargé")
        st.json(data)

st.divider()
st.header("Prédiction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        quick_ratio = st.number_input("Quick Ratio", min_value=0.0, value=5.1, step=0.1)
        roa_before_interest_and_depreciation_after_tax = st.number_input("ROA(B) before interest and depreciation after tax", min_value=0.0, value=3.5, step=0.1)
        borrowing_dependency = st.number_input("Borrowing dependency", min_value=0.0, value=0.0, step=0.1)
    with col2:
        research_and_development_expense_rate = st.number_input("Research and development expense rate", min_value=0.0, value=1.4, step=0.1)
        quick_assets_current_liability = st.number_input("Quick assets/Current Liability", min_value=0.0, value=0.2, step=0.1)

    predict_clicked = st.form_submit_button("Prédire", use_container_width=True)

if predict_clicked:
    payload = {
        "quick_ratio": quick_ratio,
        "roa_before_interest_and_depreciation_after_tax": roa_before_interest_and_depreciation_after_tax,
        "borrowing_dependency": borrowing_dependency,
        "research_and_development_expense_rate": research_and_development_expense_rate,
        "quick_assets_current_liability": quick_assets_current_liability,
    }
    data, error = call_api("POST", "/predict", json=payload)
    if error:
        st.error(f"Erreur /predict : {error}")
    else:
        prediction = data.get("prediction")
        if prediction is not None:
            st.success(f"Prédiction : {prediction}")
        st.json(data)

st.divider()
st.header("Administration du modèle")
st.warning("Le rechargement remplace le modèle en mémoire par celui actuellement ciblé par MODEL_NAME / MODEL_ALIAS.")

if st.button("Recharger le modèle /reload", type="primary", use_container_width=True):
    data, error = call_api("POST", "/reload")
    if error:
        st.error(f"Erreur /reload : {error}")
    else:
        st.success("Modèle rechargé")
        st.json(data)
