# Liste des variables à surcer vant de lancer python train.py
## Se placer dans l'environnement enterprise-risk-ai (même version mlflow sur toute la chaine des applications)
    - faire un conda activate enterprise-risk_ai  


## Creation d'un fichier secrets.sh contenant les 4 variables
```
export AWS_ACCESS_KEY_ID=Azzzzzzzzz
export AWS_SECRET_ACCESS_KEY=zzzzzzzzzzzzzzzzz
export APP_URI=https://mlflow-for-enterprise-risk-ai-ccccccccccccc.europe-west1.run.app
export EXPERIMENT_NAME=xxxxxxxxxxxxxxxxxxxx
```

## source du fichier secrets.sh pour prise en compte
```
source secrets.sh
```