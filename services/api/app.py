# app.py
from contextlib import asynccontextmanager
from http.client import HTTPException
from xml.parsers.expat import model
from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import mlflow
import pandas as pd
import os


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# Chargé une seule fois au démarrage du process
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


CLASS_NAMES = {
    0: "Pas de faillite",
    1: "Faillite",
}


def load_model():
    if MLFLOW_TRACKING_URI and MODEL_NAME and MODEL_ALIAS:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        model_info = mlflow.models.get_model_info(model_uri)
        model_id = getattr(model_info, "model_id", None)
        return model, model_uri, model_id
    raise RuntimeError(
        f"""Aucun modèle disponible : renseignez MLFLOW_TRACKING_URI et MODEL_NAME et MODEL_ALIAS
        MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}
        MODEL_NAME={MODEL_NAME}
        MODEL_ALIAS={MODEL_ALIAS}
        """
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.model_source, app.state.model_id = load_model()
    yield


app = FastAPI(title="Enterprise Risk AI", lifespan=lifespan)

class FeaturesEnterpriseRiskAi(BaseModel):
    quick_ratio: float
    roa_before_interest_and_depreciation_after_tax: float
    borrowing_dependency: float
    research_and_development_expense_rate: float
    quick_assets_current_liability: float


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "model_source": getattr(request.app.state, "model_source", None),
        "model_id": getattr(request.app.state, "model_id", None),
    }

@app.post("/predict")
def predict(request: Request, features: FeaturesEnterpriseRiskAi):

    X = pd.DataFrame([{
        "Quick Ratio": features.quick_ratio,
        "ROA(B) before interest and depreciation after tax": features.roa_before_interest_and_depreciation_after_tax,
        "Borrowing dependency": features.borrowing_dependency,
        "Research and development expense rate": features.research_and_development_expense_rate,
        "Quick Assets/Current Liability": features.quick_assets_current_liability,
    }])

    # prediction = int(request.app.state.model.predict(X)[0])

    model = request.app.state.model
    probas = model.predict_proba(X)[0]
    proba_0 = float(probas[0])
    proba_1 = float(probas[1])

    threshold = 0.50
    prediction = int(proba_1 >= threshold)
    
    # Trouver la colonne correspondant à la classe prédite
    class_index = list(request.app.state.model.classes_).index(prediction)
    probability = float(probas[class_index])

    # Vérification que la prédiction est bien dans les classes attendues
    if prediction not in range(len(CLASS_NAMES)):
        raise ValueError(
            f"Classe inattendue retournée par le modèle : {prediction}"
        )

    return {
        "prediction": CLASS_NAMES[prediction],
        "prediction_id": prediction,
        "probabilité classe prédite": round(proba_1,4),
        "probabilité_0": round(proba_0,4),
        "probabilité_1": round(proba_1,4),
        "threshold": threshold,
        "model_id": getattr(request.app.state, "model_id", None),
        "model_source": getattr(request.app.state, "model_source", None),
    }



@app.post("/reload")
def reload_model(request: Request):
    try:
        new_model, source, model_id = load_model()

        # On ne remplace l'ancien modèle que si
        # le nouveau a été chargé correctement
        request.app.state.model = new_model
        request.app.state.model_source = source
        request.app.state.model_id = model_id

        return {
            "status": "ok",
            "message": "Model reloaded",
            "model_source": source,
            "model_id": model_id,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {e}"
        )