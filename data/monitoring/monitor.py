#!/usr/bin/env python3
"""
Monitoring - EnterpriseRisk AI

Question posée : les données reçues en production ressemblent-elles
encore aux données sur lesquelles le modèle a appris ?

- Référence : le jeu d'ENTRAÎNEMENT (80 %), celui que le modèle a vu.
- Courant   : un lot "de production". Faute de vraies données de
  production, on le simule à partir du jeu de TEST :
    * scenario "normal" : le jeu de test tel quel (aucun drift attendu)
    * scenario "shock"  : un choc de liquidité appliqué au jeu de test
- Evidently compare chaque variable avec un test de Kolmogorov-Smirnov
  (p-value < 0.05 = drift) et déclare un drift global si la part de
  variables en drift atteint DRIFT_SHARE.

Pourquoi K-S et pas le test par défaut (distance de Wasserstein) ?
Le dataset contient des valeurs extrêmes (Quick Ratio jusqu'à 9e9) :
la distance de Wasserstein, normalisée par l'écart-type, est écrasée
et ne voit pas le choc. K-S compare des rangs : il est insensible
aux valeurs extrêmes.

Usage :
    # 1. Générer les lots de production simulés (une seule fois)
    python -m src.monitoring.monitor --generate

    # 2. Contrôler un lot de production reçu
    python -m src.monitoring.monitor --current-csv data/monitoring/prod_batch_2026-09_normal.csv
    python -m src.monitoring.monitor --current-csv data/monitoring/prod_batch_2026-10_shock.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data.features import SELECTED_FEATURES
from src.data.preprocess import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "monitoring"
BATCH_DIR = PROJECT_ROOT / "data" / "monitoring"

# Drift global si au moins 50 % des variables dérivent (défaut Evidently)
DRIFT_SHARE = 0.5
DRIFT_METHOD = "ks"


def simulate_liquidity_shock(X: pd.DataFrame) -> pd.DataFrame:
    """
    Scénario métier simulé : les entreprises soumises perdent 40 %
    de leur liquidité immédiate et deviennent plus dépendantes
    de l'emprunt. La rentabilité et la R&D ne bougent pas.
    """
    shocked = X.copy()
    shocked["Quick Ratio"] *= 0.6
    shocked["Quick Assets/Current Liability"] *= 0.6
    shocked["Borrowing dependency"] += 0.01
    return shocked


def build_prod_batch(X: pd.DataFrame, batch_month: str) -> pd.DataFrame:
    """
    Met un lot au format "production" : un identifiant d'entreprise,
    le mois de scoring, puis les 5 ratios reçus par l'API.
    Pas de colonne Bankrupt? : en production, l'issue n'est pas connue.
    """
    batch = X.reset_index(drop=True).copy()
    batch.insert(0, "scoring_month", batch_month)
    batch.insert(0, "company_id", [f"TW-{i:05d}" for i in X.index])
    return batch


def generate_prod_batches(X_test: pd.DataFrame) -> list:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, month, X in [
        ("normal", "2026-09", X_test),
        ("shock", "2026-10", simulate_liquidity_shock(X_test)),
    ]:
        path = BATCH_DIR / f"prod_batch_{month}_{name}.csv"
        build_prod_batch(X, month).to_csv(path, index=False)
        paths.append(path)
    return paths


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, scenario: str) -> dict:
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE, num_method=DRIFT_METHOD)])
    snapshot = report.run(current_data=current, reference_data=reference)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / f"drift_{scenario}.html"
    snapshot.save_html(str(html_path))

    # Lecture du résumé produit par Evidently
    metrics = snapshot.dict()["metrics"]
    drifted = next(m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount"))
    per_column = {
        m["metric_name"].split("column=")[1].split(",method")[0]: float(m["value"])
        for m in metrics
        if m["metric_name"].startswith("ValueDrift")
    }

    summary = {
        "scenario": scenario,
        "n_reference": len(reference),
        "n_current": len(current),
        "drifted_columns": int(drifted["value"]["count"]),
        "drifted_share": float(drifted["value"]["share"]),
        "dataset_drift": float(drifted["value"]["share"]) >= DRIFT_SHARE,
        "column_scores": per_column,
        "html_report": str(html_path),
    }
    (OUTPUT_DIR / f"drift_{scenario}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="génère les lots de production simulés")
    parser.add_argument("--current-csv", type=Path, help="lot de production à contrôler")
    parser.add_argument("--scenario", choices=["normal", "shock"], default="shock",
                        help="sans --current-csv : lot simulé à la volée")
    args = parser.parse_args()

    X_train, X_test, _, _ = prepare_dataset(CSV_PATH, features=SELECTED_FEATURES)

    if args.generate:
        for path in generate_prod_batches(X_test):
            print(f"Lot de production généré : {path}")
        return

    if args.current_csv:
        batch = pd.read_csv(args.current_csv)
        current = batch[SELECTED_FEATURES]          # on ne compare que les 5 ratios du modèle
        label = args.current_csv.stem
    else:
        current = X_test if args.scenario == "normal" else simulate_liquidity_shock(X_test)
        label = args.scenario

    s = run_drift_report(X_train, current, label)

    print("\n" + "=" * 60)
    print(f"MONITORING - lot : {s['scenario']}")
    print("=" * 60)
    print(f"Référence (train) : {s['n_reference']} lignes | Lot courant : {s['n_current']} lignes")
    print(f"Variables en drift : {s['drifted_columns']} / {len(SELECTED_FEATURES)} ({s['drifted_share']:.0%})")
    for col, p_value in s["column_scores"].items():
        flag = "DRIFT" if p_value < 0.05 else "ok"
        print(f"  - {col:52s} p-value K-S = {p_value:.3f}  [{flag}]")
    print(f"DRIFT GLOBAL : {'OUI -> analyser et envisager un retraining' if s['dataset_drift'] else 'NON'}")
    print(f"Rapport : {s['html_report']}")


if __name__ == "__main__":
    main()
