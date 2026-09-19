# EnterpriseRisk AI

Production-grade MLOps for Corporate Bankruptcy Risk Prediction.

## Objective

EnterpriseRisk AI demonstrates a complete MLOps lifecycle for
corporate bankruptcy risk scoring.

## Dataset

Taiwanese Bankruptcy Prediction - UCI Machine Learning Repository

- 6,819 companies
- 95 financial indicators
- Binary target: `Bankrupt?`
- Highly imbalanced target (~3.2% bankrupt)

## Candidate Models

- Logistic Regression
- Decision Tree
- XGBoost

The models are evaluated using metrics adapted to class imbalance,
including Recall, Precision, F1 and PR-AUC.

## MLOps Lifecycle

Data
→ Preprocessing
→ Training
→ Evaluation
→ MLflow Tracking
→ Model Selection
→ Model Registry
→ Champion
→ FastAPI
→ Docker
→ CI/CD
→ Monitoring
→ Drift Detection
→ Retraining
→ Challenger
→ Validation
→ Promotion / Rejection
→ Production

## Technology Stack

- Python
- pandas
- scikit-learn
- XGBoost
- MLflow
- FastAPI
- Pydantic
- Docker
- pytest
- GitHub Actions
- Evidently

## Repository Structure

- `data/` - raw, processed and monitoring data
- `notebooks/` - exploratory analysis
- `src/data/` - loading and preprocessing
- `src/training/` - candidate model training
- `src/evaluation/` - model evaluation and comparison
- `src/registry/` - MLflow model lifecycle
- `src/api/` - FastAPI prediction service
- `src/monitoring/` - monitoring and drift detection
- `src/retraining/` - retraining pipeline
- `tests/` - automated tests
- `.github/workflows/` - CI/CD and retraining workflows

## Environnement et fonctionnement avec les branches de dev personnalisées
### creation de l'environnement 

```
conda create -n LLMops python=3.12 -y
conda activate LLMops

contenu du fichier requirements.txt :
ragas
langchain-core>=0.2,<0.3
langchain-openai>=0.1,<0.2
openai

pandas
numpy
matplotlib

tabulate
python-dotenv
ipykernel


puis:
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```


## fonctionnement avec les branches de dev individuelles
### commit et push - sur une branche existante
git checkout dev_franck
git add -A .
git status # contrôle des fichiers inclus dans le stage qui seront commités
git commit -m "Mon super commentaire parlant"
git push

### rafraichir la branche de dev avec la main
```
git checkout main
git pull origin main

git checkout dev_franck
git merge main

git push origin dev_franck

```