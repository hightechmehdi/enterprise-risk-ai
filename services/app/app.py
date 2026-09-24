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

# Deux entreprises réelles du jeu de test (jamais vues à l'entraînement)
EXAMPLES = {
    "Entreprise saine": {
        "qr": 0.02760, "roa": 0.62514, "bd": 0.36964, "rd": 0.00011, "qa": 0.02866,
    },
    "Entreprise en faillite": {
        "qr": 0.00036, "roa": 0.46207, "bd": 0.38052, "rd": 0.00000, "qa": 0.00040,
    },
}
choice = st.radio("Exemple pré-rempli (jeu de test)", list(EXAMPLES), horizontal=True)
ex = EXAMPLES[choice]
fmt = {"min_value": 0.0, "step": 0.0001, "format": "%.5f"}

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        quick_ratio = st.number_input("Quick Ratio (liquidité immédiate)", value=ex["qr"], key=f"qr_{choice}", **fmt)
        roa_before_interest_and_depreciation_after_tax = st.number_input("ROA(B) before interest and depreciation after tax (rentabilité des actifs)", value=ex["roa"], key=f"roa_{choice}", **fmt)
        borrowing_dependency = st.number_input("Borrowing dependency (dépendance à l'emprunt)", value=ex["bd"], key=f"bd_{choice}", **fmt)
    with col2:
        research_and_development_expense_rate = st.number_input("Research and development expense rate (taux de dépenses en R&D)", value=ex["rd"], key=f"rd_{choice}", **fmt)
        quick_assets_current_liability = st.number_input("Quick assets/Current Liability (actifs liquides / dettes court terme)", value=ex["qa"], key=f"qa_{choice}", **fmt)

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
            score = data.get("probabilité_1")
            if score is not None:
                st.metric("Score de risque (non calibré)", f"{score:.2f}",
                    help=f"Seuil de décision : {data.get('threshold')}")
            if data.get("prediction_id") == 1:
                st.error(f"Prédiction : {prediction}")
            else:
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
