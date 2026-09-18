from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler


TARGET = "Bankrupt?"


# ============================================================
# 1. CHARGEMENT
# ============================================================

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Charge le fichier CSV et nettoie les noms de colonnes.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    # Supprime les espaces accidentels au début/à la fin des noms
    df.columns = df.columns.str.strip()

    print(f"Dataset chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    return df


# ============================================================
# 2. RAPPORT QUALITÉ
# ============================================================

def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Produit les informations principales permettant
    de contrôler la qualité des données.
    """

    report = {}

    report["shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }

    report["duplicates"] = int(df.duplicated().sum())

    report["missing_values"] = {
        col: int(value)
        for col, value in df.isna().sum().items()
    }

    report["dtypes"] = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    }

    # Valeurs infinies
    numeric_df = df.select_dtypes(include=np.number)

    report["infinite_values"] = {
        col: int(np.isinf(numeric_df[col]).sum())
        for col in numeric_df.columns
    }

    # Distribution cible
    if TARGET in df.columns:
        report["target_distribution"] = {
            str(k): int(v)
            for k, v in df[TARGET].value_counts().items()
        }

        report["target_distribution_pct"] = {
            str(k): round(float(v), 4)
            for k, v in df[TARGET].value_counts(normalize=True).items()
        }

    # Statistiques descriptives
    report["statistics"] = (
        numeric_df
        .describe()
        .round(4)
        .to_dict()
    )

    return report


def save_quality_report(
    report: dict,
    output_path: str | Path = "reports/data_quality_report.json"
):
    """
    Sauvegarde le rapport qualité.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False, default=str)

    print(f"Rapport qualité sauvegardé : {output_path}")


# ============================================================
# 3. VALIDATION DU SCHÉMA ET DES TYPES
# ============================================================

def validate_schema(
    df: pd.DataFrame,
    target: str = TARGET,
    expected_features: list[str] | None = None
):
    """
    Vérifie :
    - présence de la target
    - présence éventuelle des features attendues
    - target binaire
    """

    if target not in df.columns:
        raise ValueError(f"Target absente : {target}")

    if expected_features is not None:
        missing_features = [
            col for col in expected_features
            if col not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Features absentes du dataset : {missing_features}"
            )

    target_values = set(df[target].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"La target doit être binaire 0/1. Valeurs trouvées : "
            f"{target_values}"
        )


def enforce_numeric_types(
    df: pd.DataFrame,
    target: str = TARGET
) -> pd.DataFrame:
    """
    Force toutes les features au format numérique.
    Les valeurs non convertibles deviennent NaN.
    """

    df = df.copy()

    feature_columns = [
        col for col in df.columns
        if col != target
    ]

    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Target entière
    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    return df


# ============================================================
# 4. NETTOYAGE
# ============================================================

def clean_data(
    df: pd.DataFrame,
    target: str = TARGET
) -> pd.DataFrame:
    """
    Nettoyage déterministe :
    - infinis -> NaN
    - suppression des NaN
    - suppression des doublons
    - target en int
    """

    df = df.copy()

    rows_before = len(df)

    # Les infinis ne sont pas exploitables par les modèles
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_before = int(df.isna().sum().sum())

    # Dataset UCI normalement sans NaN.
    # Si on en trouve, on les retire et on les trace.
    df = df.dropna()

    duplicates_before = int(df.duplicated().sum())

    df = df.drop_duplicates()

    df[target] = df[target].astype(int)

    print("----- DATA CLEANING -----")
    print(f"Lignes initiales          : {rows_before}")
    print(f"Valeurs manquantes        : {missing_before}")
    print(f"Doublons                  : {duplicates_before}")
    print(f"Lignes après nettoyage    : {len(df)}")

    return df


# ============================================================
# 5. IDENTIFICATION DES VALEURS EXTRÊMES
# ============================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    target: str = TARGET
) -> pd.DataFrame:
    """
    Identifie les valeurs extrêmes avec la méthode IQR.

    Une valeur est considérée comme extrême si :
    x < Q1 - 1.5 * IQR
    ou
    x > Q3 + 1.5 * IQR
    """

    rows = []

    features = [
        col for col in df.columns
        if col != target
    ]

    for col in features:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (df[col] < lower)
            | (df[col] > upper)
        )

        count = int(mask.sum())

        rows.append({
            "feature": col,
            "outliers": count,
            "outlier_pct": round(
                100 * count / len(df),
                2
            ),
            "lower_bound": lower,
            "upper_bound": upper,
        })

    report = pd.DataFrame(rows)

    return report.sort_values(
        "outlier_pct",
        ascending=False
    )


# ============================================================
# 6. TRANSFORMER POUR LIMITER LES EXTREMES
# ============================================================

class QuantileClipper(
    BaseEstimator,
    TransformerMixin
):
    """
    Limite les valeurs extrêmes aux quantiles calculés
    UNIQUEMENT sur le jeu d'entraînement.

    Exemple :
    quantile 1 % et 99 %.

    Important :
    le fit ne doit jamais être fait sur le test.
    """

    def __init__(
        self,
        lower_quantile=0.01,
        upper_quantile=0.99
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        self.lower_bounds_ = X.quantile(
            self.lower_quantile
        )

        self.upper_bounds_ = X.quantile(
            self.upper_quantile
        )

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        return X.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1
        )


# ============================================================
# 7. SUPPRIMIER LES FEATURES CONSTANTES 
# ============================================================

def remove_constant_features(X_train, X_test):
    """
    Détecte les features constantes dans X_train
    et supprime les mêmes colonnes de X_train et X_test.

    La détection est faite uniquement sur le train
    pour éviter d'utiliser de l'information provenant du test.
    """

    constant_features = [
        col
        for col in X_train.columns
        if X_train[col].nunique(dropna=False) <= 1
    ]

    if constant_features:
        print(
            f"Features constantes supprimées : "
            f"{constant_features}"
        )

        X_train = X_train.drop(
            columns=constant_features
        )

        X_test = X_test.drop(
            columns=constant_features
        )

    return X_train, X_test

# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

def split_data(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.20,
    random_state: int = 42,
    features: list[str] | None = None
):

    if features is None:
        features = [
            col for col in df.columns
            if col != target
        ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
    )
    # Détection des features constantes uniquement sur le TRAIN
    # puis suppression des mêmes colonnes dans TRAIN et TEST
    X_train, X_test = remove_constant_features(
        X_train,
        X_test
    )

    print("----- TRAIN / TEST -----")
    print(f"Train : {X_train.shape}")
    print(f"Test  : {X_test.shape}")

    print("\nTarget train :")
    print(y_train.value_counts(normalize=True))

    print("\nTarget test :")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


# ============================================================
# 9. PIPELINE PREPROCESSING
# ============================================================

def build_linear_preprocessor():
    """
    Preprocessing pour Logistic Regression.

    1. limitation des valeurs extrêmes
    2. standardisation

    RobustScaler peut être particulièrement pertinent
    pour des ratios financiers contenant des valeurs extrêmes.
    """

    pipeline = Pipeline([
        (
            "outlier_clipping",
            QuantileClipper(
                lower_quantile=0.01,
                upper_quantile=0.99
            )
        ),
        (
            "scaler",
            RobustScaler()
        )
    ])

    return pipeline


def build_tree_preprocessor():
    """
    Decision Tree et XGBoost ne nécessitent pas
    de standardisation.

    On peut néanmoins appliquer le clipping
    si l'analyse montre qu'il est utile.
    """

    pipeline = Pipeline([
        (
            "outlier_clipping",
            QuantileClipper(
                lower_quantile=0.01,
                upper_quantile=0.99
            )
        )
    ])

    return pipeline


# ============================================================
# 10. PIPELINE COMPLET DE PREPROCESSING
# ============================================================

def prepare_dataset(
    csv_path: str | Path,
    features: list[str] | None = None
):

    # Chargement
    df = load_data(csv_path)

    # Rapport avant nettoyage
    report = data_quality_report(df)

    save_quality_report(
        report,
        "reports/data_quality_report_before_cleaning.json"
    )

    # Schéma
    validate_schema(
        df,
        expected_features=features
    )

    # Conversion types
    df = enforce_numeric_types(df)

    # Nettoyage
    df = clean_data(df)

    # Rapport outliers
    outlier_report = detect_outliers_iqr(df)

    Path("reports").mkdir(
        exist_ok=True
    )

    outlier_report.to_csv(
        "reports/outliers_report.csv",
        index=False
    )

    # Split
    X_train, X_test, y_train, y_test = (
        split_data(
            df,
            features=features
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# EXECUTION DIRECTE
# ============================================================

if __name__ == "__main__":

    CSV_PATH = (
        "data/raw/data.csv"
    )

    X_train, X_test, y_train, y_test = (
        prepare_dataset(CSV_PATH)
    )