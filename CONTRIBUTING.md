# Contribuer à EnterpriseRisk AI

## Organisation des branches

| Branche | Rôle |
|---|---|
| `main` | Branche de référence. Chaque merge redéploie les services (Cloud Build). Modification uniquement par pull request relue. |
| `dev_<prénom>` | Branche personnelle de chaque membre (`dev_franck`, `dev_mehdi`, `dev_olivier`, `dev_safidy`). |

## Environnement

```bash
conda create -n enterprise-risk-ai python=3.12 -y
conda activate enterprise-risk-ai
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

Les versions des dépendances sont épinglées dans `requirements.txt`. Chaque service (`services/api`, `services/app`, `services/mlflow`) a son propre `requirements.txt`, utilisé par son image Docker.

## Cycle de travail

### 1. Mettre sa branche à jour avec `main`

```bash
git checkout main
git pull origin main
git checkout dev_<prénom>
git pull
git merge main
git push
```

### 2. Développer, tester, pousser

```bash
python -m pytest -q        # tous les tests doivent passer
git add <fichiers>
git status                 # vérifier ce qui sera commité
git commit -m "Message explicite : ce qui change et pourquoi"
git push
```

Chaque push déclenche `ci.yml`, qui reconstruit l'image du service modifié.

### 3. Ouvrir une pull request

Sur GitHub : *Compare & pull request*, base `main` ← `dev_<prénom>`. Une relecture est obligatoire avant le merge.

## Checklist de pull request

- [ ] `python -m pytest -q` passe en local.
- [ ] Aucun secret, identifiant ou URL privée dans le code ou les messages de commit.
- [ ] Aucun fichier généré ajouté par erreur (voir ci-dessous).
- [ ] Les noms des 5 variables restent identiques entre `src/data/features.py`, l'API et le monitoring.
- [ ] Si la modification touche l'API ou Streamlit : vérifier `/health` et une prédiction après le déploiement.
- [ ] La documentation (`README.md`, `docs/`) est à jour si le comportement change.

## Fichiers à ne pas commiter

| Fichier | Raison |
|---|---|
| `mlflow.db`, `mlruns/` | Base MLflow locale créée quand `MLFLOW_TRACKING_URI` n'est pas défini |
| `evidently_workspace/` | Espace Evidently local, recréé à chaque contrôle |
| `reports/monitoring/*.html` | Rapports générés (plusieurs Mo), reproductibles en quelques secondes |
| `.env`, fichiers de clés | Secrets : GCP Secret Manager et GitHub Secrets uniquement |

Après un entraînement local, `git restore reports/` annule la réécriture des rapports de données.

## Conventions de code

- Les transformations qui apprennent des données (écrêtage, mise à l'échelle) sont placées dans un `Pipeline` scikit-learn, pour éviter toute fuite de données en validation croisée.
- Toute configuration dépendant de l'environnement passe par une variable d'environnement.
- Le jeu de test n'est utilisé qu'une fois, pour évaluer le modèle retenu. Aucune décision ne se prend sur lui.

## Règles propres au modèle

- Un script ne pose jamais l'alias `@champion` sur un modèle en production : seul un humain promeut.
- Ne pas lancer `retraining.yml` à la main avec l'option `champion`.
- Voir le [runbook](docs/runbook.md) pour la promotion et le rollback.
