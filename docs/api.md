# Référence de l'API

Service FastAPI (`services/api/app.py`) qui sert le modèle marqué `@champion` dans le Registry MLflow.

- **Production** : `https://fastapi-erai-463360649585.europe-west1.run.app`
- **Documentation interactive** : `/docs` (Swagger UI, avec des boutons pour tester chaque route) et `/redoc`
- **Schéma OpenAPI** : `/openapi.json`

## Fonctionnement

1. **Au démarrage**, l'API charge `models:/{MODEL_NAME}@{MODEL_ALIAS}` depuis MLflow et le garde en mémoire. Si le chargement échoue (serveur MLflow injoignable, alias absent, identifiants S3 manquants), l'application s'arrête. Sur Cloud Run, la révision précédente continue alors de servir le trafic.
2. **Chaque prédiction** utilise le modèle en mémoire : le serveur MLflow n'est pas interrogé à chaque requête.
3. **`/reload`** recharge le modèle correspondant à l'alias, par exemple après une promotion ou un rollback, sans redéployer le service.

## Configuration

| Variable | Rôle | Défaut |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Serveur MLflow | obligatoire |
| `MODEL_NAME` | Modèle dans le Registry | obligatoire (`prod-mlflow-server`) |
| `MODEL_ALIAS` | Alias chargé | obligatoire (`champion`) |
| `THRESHOLD` | Seuil de décision sur `probabilité_1` | `0.5` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Lecture du modèle dans S3 | obligatoires |

---

## `GET /health`

Indique si le service est prêt et quel modèle il sert.

**Réponse 200**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_source": "models:/prod-mlflow-server@champion",
  "model_id": "m-ebf85fae269c4a8b9c78c0a193c98433"
}
```

| Champ | Description |
|---|---|
| `model_loaded` | `true` si un modèle est en mémoire |
| `model_source` | Adresse MLflow du modèle chargé (nom + alias) |
| `model_id` | Identifiant unique du modèle dans MLflow ; il change après une promotion suivie d'un `/reload` |

---

## `POST /predict`

Calcule le score de risque d'une entreprise.

**Corps de la requête** (JSON, les 5 champs sont obligatoires ; nombres décimaux)

| Champ | Colonne du modèle | Lecture métier |
|---|---|---|
| `quick_ratio` | `Quick Ratio` | liquidité immédiate |
| `roa_before_interest_and_depreciation_after_tax` | `ROA(B) before interest and depreciation after tax` | rentabilité des actifs |
| `borrowing_dependency` | `Borrowing dependency` | dépendance à l'emprunt |
| `research_and_development_expense_rate` | `Research and development expense rate` | effort de R&D |
| `quick_assets_current_liability` | `Quick Assets/Current Liability` | couverture des dettes court terme |

Les valeurs attendues sont celles du jeu de données publié, déjà normalisées. Par exemple, un Quick Ratio typique vaut environ 0,005 à 0,03.

**Exemple : entreprise en faillite du jeu de test**

```bash
curl -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "quick_ratio": 0.00036,
    "roa_before_interest_and_depreciation_after_tax": 0.46207,
    "borrowing_dependency": 0.38052,
    "research_and_development_expense_rate": 0.0,
    "quick_assets_current_liability": 0.00040
  }'
```

**Réponse 200**

```json
{
  "prediction": "Faillite",
  "prediction_id": 1,
  "probabilité classe prédite": 0.805,
  "probabilité_0": 0.195,
  "probabilité_1": 0.805,
  "threshold": 0.5,
  "model_id": "m-…",
  "model_source": "models:/prod-mlflow-server@champion"
}
```

| Champ | Description |
|---|---|
| `prediction` | `"Faillite"` si `probabilité_1 ≥ threshold`, sinon `"Pas de faillite"` |
| `prediction_id` | 1 = faillite, 0 = pas de faillite |
| `probabilité_1` | **Score de risque** (arrondi à 4 décimales). Non calibré : il sert à classer les entreprises, pas à estimer une probabilité réelle |
| `probabilité_0` | 1 − `probabilité_1` |
| `probabilité classe prédite` | Score de la classe retenue |
| `threshold` | Seuil appliqué |
| `model_id`, `model_source` | Traçabilité : le modèle exact qui a produit la réponse |

**Exemple : entreprise saine** (score faible, « Pas de faillite »)

```json
{
  "quick_ratio": 0.02760,
  "roa_before_interest_and_depreciation_after_tax": 0.62514,
  "borrowing_dependency": 0.36964,
  "research_and_development_expense_rate": 0.00011,
  "quick_assets_current_liability": 0.02866
}
```

**Erreurs**

| Code | Cause |
|---|---|
| 422 | Champ manquant ou non numérique (validation Pydantic ; la réponse indique le champ en cause) |

Les valeurs ne sont pas contrôlées au-delà du type : une valeur hors des plages du jeu d'entraînement est acceptée et produit un score peu fiable.

---

## `POST /reload`

Recharge le modèle correspondant à `MODEL_NAME@MODEL_ALIAS`. À appeler après avoir déplacé l'alias dans MLflow (promotion ou rollback).

**Réponse 200**

```json
{
  "status": "ok",
  "message": "Model reloaded",
  "model_source": "models:/prod-mlflow-server@champion",
  "model_id": "m-…"
}
```

**Erreur 500** : le chargement a échoué (alias absent, MLflow ou S3 injoignable). L'ancien modèle reste en mémoire et continue de servir : il n'est remplacé qu'en cas de succès.

> Limite connue : `HTTPException` est actuellement importé depuis `http.client` au lieu de `fastapi`. En cas d'échec, la réponse est donc une erreur 500 générique au lieu du message détaillé. Correction : `from fastapi import FastAPI, HTTPException, Request`.

---

## Exécution locale

```bash
cd services/api
export MLFLOW_TRACKING_URI=... MODEL_NAME=prod-mlflow-server MODEL_ALIAS=champion
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
uvicorn app:app --port 8000
```

Ou en conteneur :

```bash
docker build -t enterprise-risk-api services/api
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI -e MODEL_NAME -e MODEL_ALIAS \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  enterprise-risk-api
```
