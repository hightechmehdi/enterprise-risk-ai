"""
Envoi des rapports de drift vers un workspace Evidently (interface web).

Le workspace utilisé dépend des variables d'environnement :
  EVIDENTLY_API_KEY  (+ EVIDENTLY_ORG_ID) -> Evidently Cloud
  EVIDENTLY_UI_URL                        -> serveur Evidently auto-hébergé
  sinon                                   -> dossier local evidently_workspace/

Consulter le workspace local :
    evidently ui --workspace evidently_workspace
    puis ouvrir http://localhost:8000
"""

import os
from pathlib import Path

from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import bar_plot_panel, counter_panel, line_plot_panel
from evidently.ui.workspace import CloudWorkspace, RemoteWorkspace, Workspace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_WORKSPACE_DIR = PROJECT_ROOT / "evidently_workspace"
PROJECT_NAME = "EnterpriseRisk AI - Data drift"


def get_workspace():
    """Choisit le workspace selon l'environnement (cloud, serveur, ou local)."""
    if os.getenv("EVIDENTLY_API_KEY"):
        return CloudWorkspace(token=os.environ["EVIDENTLY_API_KEY"],
                              url=os.getenv("EVIDENTLY_URL", "https://app.evidently.cloud"))
    if os.getenv("EVIDENTLY_UI_URL"):
        return RemoteWorkspace(os.environ["EVIDENTLY_UI_URL"])
    return Workspace.create(str(LOCAL_WORKSPACE_DIR))


def get_or_create_project(workspace, features: list[str]):
    """Récupère le projet par son nom, ou le crée avec son tableau de bord."""
    existing = workspace.search_project(PROJECT_NAME)
    if existing:
        return existing[0]

    project = workspace.create_project(
        PROJECT_NAME,
        description="Drift des 5 ratios du modèle de faillite (référence = train)",
        org_id=os.getenv("EVIDENTLY_ORG_ID"),
    )
    dashboard = project.dashboard
    dashboard.add_panel(counter_panel(
        title="Variables en drift (dernier lot)",
        values=[PanelMetric(metric="DriftedColumnsCount", metric_labels={"value_type": "count"})],
        size="half",
    ))
    dashboard.add_panel(counter_panel(
        title="Part de variables en drift (dernier lot)",
        values=[PanelMetric(metric="DriftedColumnsCount", metric_labels={"value_type": "share"})],
        size="half",
    ))
    dashboard.add_panel(line_plot_panel(
        title="Part de variables en drift dans le temps",
        description="Réentraînement déclenché à partir de 0.5",
        values=[PanelMetric(metric="DriftedColumnsCount", metric_labels={"value_type": "share"},
                            legend="part en drift")],
    ))
    dashboard.add_panel(bar_plot_panel(
        title="p-value K-S par variable (drift si < 0.05)",
        values=[PanelMetric(metric="ValueDrift", metric_labels={"column": col}, legend=col)
                for col in features],
    ))
    return project


def log_snapshot(snapshot, features: list[str]) -> str:
    """Ajoute un rapport au projet Evidently. Renvoie l'identifiant du projet."""
    workspace = get_workspace()
    project = get_or_create_project(workspace, features)
    workspace.add_run(project.id, snapshot, include_data=False)
    return str(project.id)