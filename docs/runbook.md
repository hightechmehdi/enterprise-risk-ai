# Runbook d'exploitation

Procédures pour faire fonctionner EnterpriseRisk AI au quotidien : déployer, entraîner, promouvoir, revenir en arrière, et diagnostiquer les incidents les plus probables.

## Règles d'or

1. **Seul un humain pose l'alias `@champion`.** Un réentraînement produit toujours un `@challenger`.
2. **Ne jamais lancer `retraining.yml` à la main avec l'option `champion`** : le modèle en production serait remplacé sans comparaison.
3. **Un changement de modèle ne nécessite aucun redéploiement** : on déplace un alias, puis on appelle `/reload`.
4. **Aucun secret dans le dépôt** : GCP Secret Manager pour Cloud Run, GitHub Secrets pour les workflows.

## Services et accès

| Élément | Où |
|---|---|
| API | Cloud Run `fastapi-erai` · `/docs`, `/health` |
| Interface | Cloud Run `streamliterai` |
| MLflow | Cloud Run `mlflow-erai` · onglets *Experiments* et *Model registry* |
| Logs | Cloud Run → service → *Journaux* (ou Cloud Logging) |
| Workflows | GitHub → *Actions* (`CI`, `Drift monitoring`, `retraining.yml`) |
| Secrets | GCP Secret Manager · GitHub → *Settings* → *Secrets and variables* |
| Modèles | S3 `mlflow-artifact-enterpriseriskai` |

---

## 1. Déployer une modification de code

1. Travailler sur sa branche personnelle, puis pousser : `ci.yml` reconstruit l'image du service modifié.
2. Ouvrir une pull request vers `main` et la faire relire.
3. Merger : Cloud Build reconstruit et redéploie le service.
4. Vérifier dans Cloud Run que la nouvelle révision est verte et reçoit 100 % du trafic, puis tester `/health`.

Si la nouvelle révision échoue, **le trafic reste sur la précédente** : le service n'est pas interrompu. Voir §6.

## 2. Réentraîner

- **Automatique** : sur drift détecté par `Drift monitoring`.
- **Manuel** : GitHub → Actions → workflow `retraining.yml` → *Run workflow*, en laissant l'option **`challenger`**.

Résultat attendu dans MLflow → *Model registry* → `prod-mlflow-server` : de nouvelles versions (aujourd'hui une par candidat), l'alias `@challenger` sur la plus récente, et `@champion` inchangé.

## 3. Promouvoir un challenger

1. **Comparer** dans MLflow les runs du champion et du challenger : PR-AUC en validation croisée, Recall.
2. **Décider.** Critères proposés, en attendant les quality gates automatiques : Recall ≥ 0,70, et PR-AUC au moins égale à celle du champion sur les mêmes données.
3. **Déplacer l'alias** : dans la version du challenger → *Aliases* → ajouter `champion`, qui quitte automatiquement l'ancienne version. Retirer `challenger` de cette version.
4. **Recharger l'API** : `POST /reload`, depuis `/docs`, depuis le bouton de Streamlit, ou :
   ```bash
   curl -X POST "$API_URL/reload"
   ```
5. **Vérifier** : `GET /health` doit renvoyer le **nouveau `model_id`**.

## 4. Revenir en arrière (rollback)

1. MLflow → version précédente → *Aliases* → ajouter `champion`.
2. `POST /reload`.
3. Vérifier que `GET /health` renvoie l'ancien `model_id`.

Durée : moins d'une minute, sans redéploiement.

## 5. Contrôler le drift à la demande

GitHub → Actions → `Drift monitoring` → *Run workflow* → scénario `shock` ou `normal`.

- **`normal`** : job vert, 1/5 variables en drift, pas de réentraînement.
- **`shock`** : 4/5, puis un job `retrain` qui se déclenche.

Le rapport Evidently est disponible dans les artefacts de l'exécution. Voir [monitoring.md](monitoring.md).

---

## 6. Diagnostiquer un incident

### L'API ne démarre pas (« container failed to start and listen on the port »)

Ce message est un **symptôme** : l'API charge le modèle **avant** d'ouvrir son port, donc tout échec de chargement produit ce message. La cause se lit dans les journaux de la révision (Cloud Run → service → *Journaux*), dans les dernières lignes avant `Application startup failed`.

| Message dans les journaux | Cause | Correction |
|---|---|---|
| `RestException: … alias champion not found` | Aucune version ne porte `@champion` (par exemple sur une base MLflow neuve) | MLflow → poser l'alias `champion` sur la version voulue, puis *Redéployer* |
| `NoCredentialsError: Unable to locate credentials` | Les identifiants AWS ne sont pas injectés dans le conteneur | Référencer les secrets `AWS_ACCESS_KEY_ID` et `AWS_SECRET_ACCESS_KEY` (voir §7) |
| `AccessDenied` … `s3:GetObject` | L'utilisateur AWS ne peut pas lire le bucket | Ajouter la politique S3 à l'utilisateur IAM (voir §7) |
| `RuntimeError: Aucun modèle disponible` | `MLFLOW_TRACKING_URI`, `MODEL_NAME` ou `MODEL_ALIAS` est vide | Compléter les variables d'environnement du service |

### Le réentraînement échoue

| Message | Cause | Correction |
|---|---|---|
| `AccessDenied` … `s3:PutObject` | L'utilisateur AWS ne peut pas écrire dans le bucket | Politique S3 (voir §7), puis *Re-run failed jobs* |
| `MLFLOW_TRACKING_URI` absent, ou écriture dans un `mlflow.db` local | Secret GitHub manquant | Ajouter le secret `MLFLOW_TRACKING_URI` au dépôt |

Un réentraînement qui échoue avant l'enregistrement du modèle ne modifie **pas** le Registry : le champion reste intact.

### Le contrôle de drift planifié ne s'est pas exécuté

GitHub ne garantit pas l'heure des exécutions planifiées ; elles peuvent être retardées ou sautées. Vérifier l'onglet *Actions*. Pour une exécution fiable, lancer le workflow à la main. En production réelle, confier la planification à un orchestrateur.

### Streamlit affiche « Erreur /predict »

Vérifier que la variable `FASTAPI_URL` du service Streamlit pointe vers l'URL de l'API, **sans `/` final**, et que `/health` de l'API répond.

---

## 7. Configuration de référence

### Secrets de l'API sur Cloud Run (Cloud Shell)

```bash
gcloud config set project <PROJECT_ID>

# Autoriser le compte de service de Cloud Run à lire les secrets
for s in ENTERPRISERISKAI_AWS_ACCESS_KEY_ID ENTERPRISERISKAI_AWS_SECRET_ACCESS_KEY; do
  gcloud secrets add-iam-policy-binding $s \
    --member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done

# Brancher les secrets sur l'API (crée une nouvelle révision)
gcloud run services update fastapi-erai --region europe-west1 \
  --update-secrets=AWS_ACCESS_KEY_ID=ENTERPRISERISKAI_AWS_ACCESS_KEY_ID:latest,AWS_SECRET_ACCESS_KEY=ENTERPRISERISKAI_AWS_SECRET_ACCESS_KEY:latest
```

Si les variables existent déjà en texte simple, ajouter `--remove-env-vars=AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY` à la dernière commande.

### Politique IAM AWS de l'utilisateur applicatif

Droits limités au seul bucket d'artefacts :

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": "arn:aws:s3:::mlflow-artifact-enterpriseriskai"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::mlflow-artifact-enterpriseriskai/*"
    }
  ]
}
```

### Cohérence entre environnements

L'API, Streamlit, les workflows et le monitoring doivent tous pointer vers **le même serveur MLflow** : même `MLFLOW_TRACKING_URI` dans Cloud Run et dans les secrets GitHub. Un changement de base MLflow impose de reposer l'alias `champion` avant de redéployer l'API.
