#!/usr/bin/env python3

"""
Sélection automatique des features - EnterpriseRisk AI

Principe :
1. Split TRAIN / TEST stratifié
2. Travail uniquement sur TRAIN
3. Mesure de l'impact de chaque feature par Permutation Importance
   avec la PR-AUC comme métrique
4. Conservation des features dont l'impact est positif et stable
5. Suppression des fortes redondances entre features
6. La variable la plus prédictive est conservée lorsqu'une paire
   est trop corrélée

Le nombre final de features n'est PAS fixé à l'avance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Bankrupt?"

RANDOM_STATE = 42
TEST_SIZE = 0.20

N_SPLITS = 5
N_REPEATS = 5

# Au-delà de ce seuil, deux variables sont considérées
# comme fortement redondantes.
CORRELATION_THRESHOLD = 0.85


# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "data.csv"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


# ============================================================
# 1. CHARGEMENT
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            f"Cible '{TARGET}' absente du dataset."
        )

    X = df.drop(
        columns=[TARGET]
    ).select_dtypes(
        include=[np.number]
    )

    y = df[TARGET].astype(int)

    print(
        f"Dataset : {len(df)} lignes, "
        f"{X.shape[1]} features."
    )

    return X, y


# ============================================================
# 2. TRAIN / TEST
# ============================================================

def split_data(X, y):

    (
        X_train,
        X_test,
        y_train,
        y_test,

    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        f"Train : {X_train.shape}"
    )

    print(
        f"Test  : {X_test.shape}"
    )

    print(
        f"Taux faillite train : "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Taux faillite test  : "
        f"{y_test.mean() * 100:.2f}%"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# 3. NETTOYAGE DU TRAIN
# ============================================================

def prepare_train_data(X_train):

    X_train = X_train.copy()

    # --------------------------------------------------------
    # Infini -> NaN
    # --------------------------------------------------------

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Imputation par médiane calculée uniquement sur TRAIN
    # --------------------------------------------------------

    medians = X_train.median()

    X_train = X_train.fillna(
        medians
    )

    # --------------------------------------------------------
    # Suppression des variables constantes
    # --------------------------------------------------------

    constant_features = [
        feature
        for feature in X_train.columns
        if X_train[feature].nunique() <= 1
    ]

    if constant_features:

        print(
            "\nVariables constantes supprimées :"
        )

        for feature in constant_features:
            print(
                f"  - {feature}"
            )

        X_train = X_train.drop(
            columns=constant_features
        )

    return (
        X_train,
        constant_features,
    )


# ============================================================
# 4. PERMUTATION IMPORTANCE EN CROSS-VALIDATION
# ============================================================

def compute_permutation_importance_cv(
    X_train,
    y_train,
):

    """
    Pour chaque fold :

    - entraînement Random Forest
    - mesure de la PR-AUC sur validation
    - permutation de chaque feature
    - mesure de la baisse de PR-AUC

    Une feature importante provoque une baisse de performance
    lorsqu'on détruit son information.
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_importances = []

    for fold, (
        train_idx,
        validation_idx,
    ) in enumerate(
        cv.split(
            X_train,
            y_train
        ),
        start=1,
    ):

        print(
            f"\nFold {fold}/{N_SPLITS}"
        )

        X_fold_train = (
            X_train.iloc[
                train_idx
            ]
        )

        X_fold_validation = (
            X_train.iloc[
                validation_idx
            ]
        )

        y_fold_train = (
            y_train.iloc[
                train_idx
            ]
        )

        y_fold_validation = (
            y_train.iloc[
                validation_idx
            ]
        )

        # ----------------------------------------------------
        # Modèle utilisé pour mesurer l'utilité des features
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )

        model.fit(
            X_fold_train,
            y_fold_train
        )

        # ----------------------------------------------------
        # Permutation Importance
        # ----------------------------------------------------

        result = permutation_importance(
            model,
            X_fold_validation,
            y_fold_validation,
            scoring="average_precision",
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )

        fold_result = pd.DataFrame({
            "feature":
                X_train.columns,

            "importance":
                result.importances_mean,

            "fold":
                fold,
        })

        all_importances.append(
            fold_result
        )

    # ========================================================
    # AGRÉGATION DES 5 FOLDS
    # ========================================================

    all_importances = pd.concat(
        all_importances,
        ignore_index=True,
    )

    summary = (
        all_importances
        .groupby("feature")["importance"]
        .agg(
            mean_importance="mean",
            std_importance="std",
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Score conservateur
    # --------------------------------------------------------
    #
    # Impact moyen - variabilité
    #
    # Si ce score reste > 0 :
    # l'impact est considéré comme suffisamment stable.
    # --------------------------------------------------------

    summary[
        "stable_importance"
    ] = (
        summary["mean_importance"]
        - summary["std_importance"]
    )

    summary = summary.sort_values(
        "mean_importance",
        ascending=False,
    ).reset_index(drop=True)

    summary["rank"] = (
        summary.index + 1
    )

    return summary


# ============================================================
# 5. FEATURES AYANT UN IMPACT REEL
# ============================================================

def select_useful_features(
    importance_df,
):

    useful = importance_df[
        importance_df[
            "stable_importance"
        ] > 0
    ].copy()

    print(
        "\n========================================"
    )

    print(
        "FEATURES AVEC IMPACT STABLE POSITIF"
    )

    print(
        "========================================"
    )

    print(
        f"{len(useful)} features retenues "
        "avant contrôle des corrélations."
    )

    return useful


# ============================================================
# 6. SUPPRESSION DES REDONDANCES
# ============================================================

def remove_correlated_features(
    useful_features,
    X_train,
):

    """
    Les features sont parcourues de la plus prédictive
    à la moins prédictive.

    Une feature est gardée uniquement si elle n'est pas
    trop corrélée avec une feature déjà sélectionnée.

    Ainsi, dans un groupe de variables très corrélées,
    on conserve la plus prédictive.
    """

    correlation_matrix = (
        X_train.corr().abs()
    )

    selected = []

    rejected = []

    for _, row in (
        useful_features.iterrows()
    ):

        candidate = row["feature"]

        too_correlated = False

        for selected_feature in selected:

            correlation = (
                correlation_matrix.loc[
                    candidate,
                    selected_feature
                ]
            )

            if (
                correlation
                >= CORRELATION_THRESHOLD
            ):

                rejected.append({
                    "feature":
                        candidate,

                    "correlated_with":
                        selected_feature,

                    "correlation":
                        correlation,

                    "mean_importance":
                        row[
                            "mean_importance"
                        ],
                })

                too_correlated = True

                break

        if not too_correlated:

            selected.append(
                candidate
            )

    return (
        selected,
        pd.DataFrame(rejected),
    )


# ============================================================
# 7. RAPPORT FINAL
# ============================================================

def build_final_report(
    selected,
    importance_df,
):

    final_report = (
        importance_df[
            importance_df[
                "feature"
            ].isin(selected)
        ]
        .copy()
    )

    final_report = (
        final_report
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    final_report[
        "final_rank"
    ] = (
        final_report.index + 1
    )

    return final_report


# ============================================================
# 8. MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 70
    )

    print(
        "ENTERPRISERISK AI - FEATURE SELECTION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Split AVANT sélection
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,

    ) = split_data(
        X,
        y
    )

    # --------------------------------------------------------
    # Nettoyage TRAIN uniquement
    # --------------------------------------------------------

    (
        X_train,
        constant_features,

    ) = prepare_train_data(
        X_train
    )

    # --------------------------------------------------------
    # Permutation Importance
    # --------------------------------------------------------

    importance_df = (
        compute_permutation_importance_cv(
            X_train,
            y_train,
        )
    )

    # --------------------------------------------------------
    # Sauvegarde de toutes les importances
    # --------------------------------------------------------

    importance_df.to_csv(
        REPORTS_DIR
        / "feature_permutation_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Features réellement utiles
    # --------------------------------------------------------

    useful_features = (
        select_useful_features(
            importance_df
        )
    )

    # --------------------------------------------------------
    # Suppression des fortes corrélations
    # --------------------------------------------------------

    (
        selected_features,
        rejected_correlations,

    ) = remove_correlated_features(
        useful_features,
        X_train,
    )

    # --------------------------------------------------------
    # Rapport final
    # --------------------------------------------------------

    final_report = (
        build_final_report(
            selected_features,
            importance_df,
        )
    )

    # --------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FEATURES FINALES"
    )

    print(
        "========================================"
    )

    print(
        f"\nNombre final : "
        f"{len(final_report)}"
    )

    for _, row in (
        final_report.iterrows()
    ):

        print(
            f"\n{int(row['final_rank']):2d}. "
            f"{row['feature']}"
        )

        print(
            f"    Impact PR-AUC moyen : "
            f"{row['mean_importance']:.5f}"
        )

        print(
            f"    Ecart-type          : "
            f"{row['std_importance']:.5f}"
        )

        print(
            f"    Impact stable       : "
            f"{row['stable_importance']:.5f}"
        )

    # --------------------------------------------------------
    # Features rejetées pour corrélation
    # --------------------------------------------------------

    if not rejected_correlations.empty:

        print(
            "\n========================================"
        )

        print(
            "FEATURES REJETEES POUR REDONDANCE"
        )

        print(
            "========================================"
        )

        for _, row in (
            rejected_correlations.iterrows()
        ):

            print(
                f"\n- {row['feature']}"
            )

            print(
                f"  trop corrélée avec : "
                f"{row['correlated_with']}"
            )

            print(
                f"  corrélation : "
                f"{row['correlation']:.3f}"
            )

    # --------------------------------------------------------
    # Sauvegardes
    # --------------------------------------------------------

    final_report.to_csv(
        REPORTS_DIR
        / "selected_features_final.csv",
        index=False,
    )

    rejected_correlations.to_csv(
        REPORTS_DIR
        / "rejected_correlated_features.csv",
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "RAPPORTS GENERES"
    )

    print(
        "========================================"
    )

    print(
        "\nreports/feature_permutation_importance.csv"
    )

    print(
        "reports/selected_features_final.csv"
    )

    print(
        "reports/rejected_correlated_features.csv"
    )

    print(
        "\nLe TEST n'a pas été utilisé "
        "pour la sélection des features."
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    main()