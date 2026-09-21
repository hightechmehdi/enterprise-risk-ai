#!/usr/bin/env python3

"""
Training pipeline - EnterpriseRisk AI

Rôle de ce fichier :
1. Récupérer les features officielles
2. Préparer TRAIN / TEST
3. Charger les modèles candidats
4. Faire le tuning uniquement sur TRAIN
5. Comparer les modèles par cross-validation
6. Sélectionner le meilleur candidat
7. Evaluer ce candidat une seule fois sur TEST
8. Logger les résultats dans MLflow
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import os

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from src.data.features import (
    TARGET,
    SELECTED_FEATURES,
)

from src.data.preprocess import (
    prepare_dataset,
    build_linear_preprocessor,
    build_tree_preprocessor,
)

from src.training.models import get_model_configs

from src.evaluation.evaluate import evaluate_model


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
EXPERIMENT_NAME=os.environ["EXPERIMENT_NAME"]
MODEL_NAME = "prod-mlflow-server"

PRIMARY_METRIC = "average_precision"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "data.csv"
)

MLFLOW_EXPERIMENT = "EnterpriseRisk_Model_Selection"


# ============================================================
# CROSS-VALIDATION
# ============================================================

def build_cv():

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ============================================================
# CONSTRUCTION DU PIPELINE
# ============================================================

def build_pipeline(
    model_name,
    estimator,
):

    """
    Associe le preprocessing adapté au modèle.

    Logistic Regression :
        clipping + RobustScaler

    Decision Tree / XGBoost :
        preprocessing arbre
        sans scaling
    """

    if model_name == "logistic_regression":

        preprocessor = (
            build_linear_preprocessor()
        )

    else:

        preprocessor = (
            build_tree_preprocessor()
        )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            estimator,
        ),
    ])

    return pipeline


# ============================================================
# TUNING D'UN MODELE
# ============================================================

def tune_model(
    model_name,
    config,
    X_train,
    y_train,
):

    """
    Lance GridSearchCV ou RandomizedSearchCV
    uniquement sur TRAIN.
    """

    pipeline = build_pipeline(
        model_name,
        config["estimator"],
    )

    cv = build_cv()

    scoring = {
        "pr_auc":
            "average_precision",

        "recall":
            "recall",

        "precision":
            "precision",

        "f1":
            "f1",

        "roc_auc":
            "roc_auc",
    }

    search_type = config.get(
        "search_type",
        "grid",
    )

    if search_type == "random":

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=
                config["params"],
            n_iter=config.get(
                "n_iter",
                20,
            ),
            scoring=scoring,
            refit="pr_auc",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )

    else:

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=
                config["params"],
            scoring=scoring,
            refit="pr_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

    search.fit(
        X_train,
        y_train,
    )

    return search


# ============================================================
# EXTRACTION DES METRIQUES CV
# ============================================================

def get_cv_metrics(search):

    """
    Récupère les métriques du meilleur jeu
    d'hyperparamètres trouvé par la CV.
    """

    index = search.best_index_

    results = search.cv_results_

    metrics = {

        "cv_pr_auc":
            results[
                "mean_test_pr_auc"
            ][index],

        "cv_recall":
            results[
                "mean_test_recall"
            ][index],

        "cv_precision":
            results[
                "mean_test_precision"
            ][index],

        "cv_f1":
            results[
                "mean_test_f1"
            ][index],

        "cv_roc_auc":
            results[
                "mean_test_roc_auc"
            ][index],
    }

    return metrics


# ============================================================
# TRAINING DE TOUS LES MODELES
# ============================================================

def train_candidates(
    X_train,
    y_train,
):

    model_configs = (
        get_model_configs(
            y_train=y_train
        )
    )

    results = {}

    for model_name, config in (
        model_configs.items()
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"TRAINING : {model_name}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Nouveau run MLflow
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=model_name
        ):

            # ------------------------------------------------
            # Tuning
            # ------------------------------------------------

            search = tune_model(
                model_name,
                config,
                X_train,
                y_train,
            )

            best_model = (
                search.best_estimator_
            )

            cv_metrics = (
                get_cv_metrics(
                    search
                )
            )

            # ------------------------------------------------
            # Affichage
            # ------------------------------------------------

            print(
                "\nMeilleurs paramètres :"
            )

            print(
                search.best_params_
            )

            print(
                "\nMétriques CV :"
            )

            for metric, value in (
                cv_metrics.items()
            ):

                print(
                    f"{metric}: "
                    f"{value:.4f}"
                )

            # ------------------------------------------------
            # MLflow
            # ------------------------------------------------

            mlflow.log_param(
                "model_name",
                model_name,
            )

            mlflow.log_param(
                "n_features",
                len(
                    SELECTED_FEATURES
                ),
            )

            mlflow.log_param(
                "features",
                str(
                    SELECTED_FEATURES
                ),
            )

            for param, value in (
                search.best_params_.items()
            ):

                mlflow.log_param(
                    param,
                    value,
                )

            for metric, value in (
                cv_metrics.items()
            ):

                mlflow.log_metric(
                    metric,
                    float(value),
                )

            # Exemple représentatif pour la signature MLflow
            input_example = X_train.head(2)

            y_example = best_model.predict(input_example)

            signature = infer_signature(
                input_example,
                y_example,
            )

            mlflow.sklearn.log_model(
                best_model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=MODEL_NAME,
                serialization_format="cloudpickle",
            )

            # ------------------------------------------------
            # Conservation du résultat
            # ------------------------------------------------

            results[model_name] = {

                "model":
                    best_model,

                "search":
                    search,

                "metrics":
                    cv_metrics,
            }

    return results


# ============================================================
# CHOIX DU MEILLEUR CANDIDAT
# ============================================================

def select_best_candidate(
    results,
):

    """
    Sélection selon la PR-AUC moyenne
    de cross-validation.
    """

    best_name = max(
        results,
        key=lambda name:
            results[name][
                "metrics"
            ][
                "cv_pr_auc"
            ]
    )

    return (
        best_name,
        results[best_name],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ENTERPRISERISK AI - TRAINING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 1. MLflow
    # --------------------------------------------------------

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    # --------------------------------------------------------
    # 2. Préparation des données
    # --------------------------------------------------------

    print(
        "\n1. PREPARATION DES DONNEES"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,

    ) = prepare_dataset(

        CSV_PATH,

        features=
            SELECTED_FEATURES,
    )

    print(
        "\nFeatures utilisées :"
    )

    for feature in (
        SELECTED_FEATURES
    ):

        print(
            f"  - {feature}"
        )

    print(
        f"\nX_train : "
        f"{X_train.shape}"
    )

    print(
        f"X_test  : "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # 3. Training + tuning
    # --------------------------------------------------------

    print(
        "\n2. TRAINING DES CANDIDATS"
    )

    results = train_candidates(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 4. Sélection du meilleur candidat
    # --------------------------------------------------------

    (
        best_name,
        best_result,

    ) = select_best_candidate(
        results
    )

    best_model = (
        best_result["model"]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MEILLEUR CANDIDAT SUR TRAIN / CV"
    )

    print(
        "=" * 70
    )

    print(
        f"\nModèle : {best_name}"
    )

    print(
        f"PR-AUC CV : "
        f"{best_result['metrics']['cv_pr_auc']:.4f}"
    )

    # --------------------------------------------------------
    # 5. Evaluation finale sur TEST
    # --------------------------------------------------------

    print(
        "\n3. EVALUATION FINALE SUR TEST"
    )

    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    print(
        "\nRésultats TEST :"
    )

    for metric, value in (
        test_metrics.items()
    ):

        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # 6. Résumé
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING TERMINE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCandidat retenu : "
        f"{best_name}"
    )

    print(
        "\nLe TEST n'a été utilisé "
        "qu'après la sélection du meilleur "
        "modèle sur TRAIN/CV."
    )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    last_version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, "champion", last_version)
    print(f"Modèle '{MODEL_NAME}' version {last_version} enregistré avec l'alias @champion")

# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()