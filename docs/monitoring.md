# Monitoring et détection de drift

## 1. Ce que l'on surveille, et pourquoi

On ne sait qu'une entreprise a fait faillite que des mois plus tard : la performance réelle du modèle ne peut pas être mesurée au moment où il prédit. On surveille donc ce qui est observable tout de suite : **les données reçues ressemblent-elles encore à celles sur lesquelles le modèle a appris ?** C'est le **data drift**.

| Type | Question | État |
|---|---|---|
| **Data drift** | La distribution des 5 variables d'entrée a-t-elle changé ? | En place (Evidently) |
| **Performance drift** | Le modèle se trompe-t-il davantage qu'au moment de sa validation ? | Prévu : nécessite de journaliser les prédictions puis de récupérer les vraies issues |

Un drift n'est pas une panne : c'est un signal. Il déclenche un réentraînement dont le résultat reste un **challenger**. Le modèle en production n'est jamais remplacé automatiquement.

## 2. Méthode

- **Référence** : le jeu d'entraînement (5 455 entreprises), c'est-à-dire ce que le modèle a vu.
- **Lot courant** : un lot de production (1 364 entreprises dans les scénarios fournis). Seules les 5 variables du modèle sont comparées : `company_id` et `scoring_month` sont ignorés.
- **Test par variable** : Kolmogorov-Smirnov (Evidently, `num_method="ks"`). Une variable est en drift si sa **p-value est inférieure à 0,05**.
- **Décision globale** : **drift si au moins 50 % des variables dérivent**, soit au moins 3 sur 5 (`drift_share=0.5`).

### Pourquoi Kolmogorov-Smirnov

Le test utilisé par défaut par Evidently sur ce volume de données, la distance de Wasserstein normalisée, **ne détectait pas le choc simulé**. Les valeurs extrêmes du dataset (Quick Ratio jusqu'à environ 9 × 10⁹) gonflent l'écart-type qui sert à normaliser la distance, et masquent le décalage. Kolmogorov-Smirnov compare des rangs : il est insensible aux valeurs extrêmes.

### Pourquoi un seuil de 50 %

Au seuil de 5 %, une variable sur vingt « dérive » par pur hasard. Le lot normal en donne un exemple : ROA(B) y ressort (p = 0,043) alors que rien n'a changé. Décider sur une seule variable produirait de fausses alertes.

## 3. Lots de production simulés

Faute de flux de production réel, deux lots sont construits à partir du **jeu de test**, c'est-à-dire des entreprises jamais vues à l'entraînement :

| Fichier | Contenu | Résultat attendu |
|---|---|---|
| `data/monitoring/prod_batch_2026-09_normal.csv` | Les entreprises telles quelles | 1 variable sur 5 en drift, **pas d'alerte** |
| `data/monitoring/prod_batch_2026-10_shock.csv` | **Choc de liquidité** : Quick Ratio × 0,6, Quick Assets/Current Liability × 0,6, Borrowing dependency + 0,01 | 4 variables sur 5 en drift, **drift global** |

Chaque ligne contient `company_id`, `scoring_month` et les 5 ratios, **sans colonne `Bankrupt?`** : en production, l'issue n'est pas connue. Le choc sur la dépendance à l'emprunt (+0,01) représente environ 0,7 écart-type de cette variable, qui varie très peu.

### Résultats de référence

| Variable | Choc | p-value · lot normal | p-value · lot choqué |
|---|---|---|---|
| Quick Ratio | × 0,6 | 0,288 | < 0,001 |
| ROA(B) before interest and depreciation after tax | aucun | 0,043 | 0,043 |
| Borrowing dependency | + 0,01 | 0,956 | < 0,001 |
| Research and development expense rate | aucun | 0,971 | 0,971 |
| Quick Assets/Current Liability | × 0,6 | 0,514 | < 0,001 |
| **Variables en drift** | | **1 / 5 → pas d'alerte** | **4 / 5 → drift global** |

Ces résultats sont vérifiés par `tests/test_monitoring.py`.

## 4. Utilisation

Depuis la racine du dépôt :

```bash
export PYTHONPATH=.

# Générer les deux lots simulés
python -m src.monitoring.monitor --generate

# Contrôler un lot
python -m src.monitoring.monitor --current-csv data/monitoring/prod_batch_2026-10_shock.csv

# Contrôler sans envoyer le rapport à Evidently UI
python -m src.monitoring.monitor --current-csv <lot.csv> --no-ui
```

**Sorties** :
- dans le terminal : p-value par variable, nombre de variables en drift, verdict ;
- `reports/monitoring/drift_<lot>.html` : rapport Evidently, à ouvrir dans un navigateur ;
- `reports/monitoring/drift_<lot>.json` : résumé exploitable par un script ;
- dans GitHub Actions : `drift_detected`, `drift_share` et `drifted_columns` sont transmis au workflow (`GITHUB_OUTPUT`).

## 5. Interface Evidently

Chaque exécution envoie son rapport dans le projet Evidently **« EnterpriseRisk AI - Data drift »**. L'espace est local (`evidently_workspace/`) par défaut, ou distant si `EVIDENTLY_UI_URL` ou `EVIDENTLY_API_KEY` est défini.

```bash
evidently ui --workspace evidently_workspace --port 8001
# puis http://localhost:8001
```

- **Dashboard** : compteur et évolution de la part de variables en drift, p-values par variable.
- **Reports** : un rapport par lot. *View* ouvre le détail par variable. Pour visualiser le choc, le graphique de *Borrowing dependency* est le plus lisible ; celui du Quick Ratio est écrasé par les valeurs extrêmes.

Chaque exécution ajoute un rapport. Pour repartir de zéro : arrêter l'interface, supprimer `evidently_workspace/`, puis relancer les contrôles.

## 6. Automatisation — `.github/workflows/monitoring.yml`

| Étape | Détail |
|---|---|
| Déclenchement | Planifié chaque jour à 10:00 (Europe/Paris) sur le lot normal ; ou manuel, avec le choix du scénario (`shock` par défaut) |
| 1. Tests | `pytest tests/test_monitoring.py` |
| 2. Lots | `monitor.py --generate` |
| 3. Détection | `monitor.py --current-csv <lot>` → sorties `drift_detected`, `drift_share`, `drifted_columns` |
| 4. Traçabilité | Résumé dans l'onglet Actions ; rapports publiés en artefact (conservés 30 jours) |
| 5. Réentraînement | Si `drift_detected == true` : appel de `retraining.yml` avec l'alias `challenger` |

- Le lancement planifié contrôle le lot **normal** : il ne doit pas déclencher de réentraînement. Une exécution verte sans réentraînement est le comportement attendu.
- GitHub peut retarder ou sauter une exécution planifiée. Pour une démonstration, utiliser le lancement manuel.

## 7. Limites

- Les lots sont simulés. Le réentraînement déclenché utilise les mêmes données d'entraînement : il démontre le mécanisme, pas l'adaptation à de nouvelles données.
- Kolmogorov-Smirnov devient très sensible sur de grands lots : un écart minime peut y devenir significatif. En production, il faudrait compléter la p-value par une mesure de l'ampleur du décalage.
- Le seuil de 50 % est le réglage par défaut d'Evidently ; il est à calibrer sur un historique réel.
- La distribution des scores produits par le modèle n'est pas encore surveillée. C'est une extension simple, qui ne nécessite pas de labels.
