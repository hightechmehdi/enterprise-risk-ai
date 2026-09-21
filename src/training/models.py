"""
Model configurations - EnterpriseRisk AI

Ce fichier définit :
- les modèles candidats ;
- leurs hyperparamètres à tester ;
- la stratégie de recherche utilisée par train.py.

Aucun entraînement n'est lancé ici.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


def compute_scale_pos_weight(y_train):
    """
    Calcule le poids de la classe positive pour XGBoost.

    scale_pos_weight = nombre classe 0 / nombre classe 1
    """

    counts = y_train.value_counts()

    n_negative = counts.get(0, 0)
    n_positive = counts.get(1, 0)

    if n_positive == 0:
        raise ValueError(
            "Impossible de calculer scale_pos_weight : "
            "aucune observation positive dans y_train."
        )

    return n_negative / n_positive


def get_model_configs(y_train):
    """
    Retourne la configuration des modèles utilisée par train.py.

    Pour chaque modèle :
    - estimator   : modèle sklearn / XGBoost
    - search_type : grid ou random
    - params      : hyperparamètres à explorer
    """

    scale_pos_weight = compute_scale_pos_weight(y_train)

    print(
        f"XGBoost scale_pos_weight = "
        f"{scale_pos_weight:.2f}"
    )

    return {

        # ====================================================
        # LOGISTIC REGRESSION
        # ====================================================

        "logistic_regression": {

            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),

            "search_type": "grid",

            "params": {
                "model__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                ],

                "model__solver": [
                    "liblinear",
                    "lbfgs",
                ],
            },
        },


        # ====================================================
        # DECISION TREE
        # ====================================================

        "decision_tree": {

            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

            "search_type": "grid",

            "params": {
                "model__max_depth": [
                    3,
                    5,
                    7,
                    10,
                    None,
                ],

                "model__min_samples_split": [
                    2,
                    5,
                    10,
                    20,
                ],

                "model__min_samples_leaf": [
                    1,
                    2,
                    5,
                    10,
                ],

                "model__criterion": [
                    "gini",
                    "entropy",
                ],
            },
        },


        # ====================================================
        # XGBOOST
        # ====================================================

        "xgboost": {

            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=1,
            ),

            "search_type": "random",

            "n_iter": 25,

            "params": {
                "model__n_estimators": [
                    100,
                    200,
                    300,
                    500,
                ],

                "model__max_depth": [
                    2,
                    3,
                    4,
                    5,
                    6,
                ],

                "model__learning_rate": [
                    0.01,
                    0.03,
                    0.05,
                    0.1,
                    0.2,
                ],

                "model__subsample": [
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],

                "model__colsample_bytree": [
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],

                "model__min_child_weight": [
                    1,
                    3,
                    5,
                    10,
                ],
            },
        },
    }