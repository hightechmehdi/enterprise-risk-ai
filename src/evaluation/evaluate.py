#!/usr/bin/env python3

"""
Evaluation des modèles - EnterpriseRisk AI

Ce fichier calcule les principales métriques
pour notre problème de classification déséquilibrée.

Métrique principale :
- PR-AUC / Average Precision

Métriques complémentaires :
- Recall
- Precision
- F1
- ROC-AUC
"""

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# EVALUATION PRINCIPALE
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold: float = 0.50,
):
    """
    Evalue un modèle déjà entraîné.

    Parameters
    ----------
    model :
        Modèle sklearn / Pipeline déjà fit.

    X :
        Features à évaluer.

    y :
        Vraies classes 0/1.

    threshold :
        Seuil utilisé pour transformer le score
        probabiliste en prédiction 0/1.

        Défaut : 0.50

    Returns
    -------
    dict contenant uniquement des métriques numériques.
    """

    # --------------------------------------------------------
    # 1. SCORE CONTINU
    # --------------------------------------------------------

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise ValueError(
            "Le modèle doit fournir predict_proba()."
        )

    y_score = (
        model
        .predict_proba(X)[:, 1]
    )

    # --------------------------------------------------------
    # 2. SCORE -> CLASSE
    # --------------------------------------------------------

    y_pred = (
        y_score >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # 3. METRIQUES
    # --------------------------------------------------------

    metrics = {

        "pr_auc":
            average_precision_score(
                y,
                y_score,
            ),

        "recall":
            recall_score(
                y,
                y_pred,
                zero_division=0,
            ),

        "precision":
            precision_score(
                y,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                y_pred,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y,
                y_score,
            ),

        "threshold":
            threshold,
    }

    return metrics


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

def get_confusion_matrix(
    model,
    X,
    y,
    threshold: float = 0.50,
):
    """
    Retourne les quatre composantes :

    TN = True Negative
    FP = False Positive
    FN = False Negative
    TP = True Positive
    """

    y_score = (
        model
        .predict_proba(X)[:, 1]
    )

    y_pred = (
        y_score >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            y_pred,
        ).ravel()
    )

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# ANALYSE DES SEUILS
# ============================================================

def evaluate_thresholds(
    model,
    X,
    y,
    thresholds=None,
):
    """
    Compare Recall / Precision / F1 pour plusieurs
    seuils de décision.

    Utile après la sélection du modèle pour choisir
    un seuil métier approprié.
    """

    if thresholds is None:

        thresholds = [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]

    y_score = (
        model
        .predict_proba(X)[:, 1]
    )

    results = []

    for threshold in thresholds:

        y_pred = (
            y_score >= threshold
        ).astype(int)

        results.append({

            "threshold":
                threshold,

            "recall":
                recall_score(
                    y,
                    y_pred,
                    zero_division=0,
                ),

            "precision":
                precision_score(
                    y,
                    y_pred,
                    zero_division=0,
                ),

            "f1":
                f1_score(
                    y,
                    y_pred,
                    zero_division=0,
                ),
        })

    return results