# Fiche du modèle — EnterpriseRisk AI

Cette fiche décrit le modèle servi en production : ce qu'il fait, sur quelles données il a appris, comment il a été choisi et évalué, et ce qu'il ne faut pas lui demander.

## 1. Identité

| | |
|---|---|
| **Nom dans le Registry** | `prod-mlflow-server` |
| **Alias servi** | `@champion`, posé par un humain |
| **Algorithme** | XGBoost (`XGBClassifier`, `tree_method="hist"`), dans un `Pipeline` scikit-learn avec écrêtage des valeurs extrêmes (`QuantileClipper`) |
| **Entrées** | 5 ratios financiers (voir §3) |
| **Sortie** | Score de risque dans [0, 1] (`probabilité_1`) et décision au seuil `THRESHOLD` (0,5 par défaut) |
| **Expérience MLflow** | `EnterpriseRisk_Model_Selection` |
| **Code** | `src/training/train.py`, `src/training/models.py` |

## 2. Usage prévu

- **Usage visé** : aider une équipe risque à **prioriser** les entreprises à examiner. Le score sert à classer ; la décision finale reste humaine.
- **Hors périmètre** : décision automatique d'octroi ou de refus de crédit ; lecture du score comme une probabilité de faillite ; application à d'autres pays, périodes ou secteurs sans nouvel entraînement ni validation.

## 3. Données

- **Source** : UCI *Taiwanese Bankruptcy Prediction* (Taiwan Economic Journal, 1999-2009).
- **Volume** : 6 819 entreprises, 95 ratios, cible `Bankrupt?` (220 faillites, soit 3,23 %). Aucune valeur manquante, aucun doublon.
- **Particularités** : ratios déjà normalisés par l'éditeur (lecture relative, pas en unités comptables) ; certaines colonnes contiennent des valeurs extrêmes aberrantes (jusqu'à environ 10⁹–10¹⁰). Les noms de colonnes du CSV commencent par un espace, retiré au chargement.
- **Découpage** : 80 % entraînement (5 455 entreprises, 176 faillites) / 20 % test (1 364, 44 faillites), stratifié, `random_state=42`.

### Variables retenues

| Colonne (nom exact) | Champ API | Lecture métier |
|---|---|---|
| `Quick Ratio` | `quick_ratio` | liquidité immédiate |
| `ROA(B) before interest and depreciation after tax` | `roa_before_interest_and_depreciation_after_tax` | rentabilité des actifs |
| `Borrowing dependency` | `borrowing_dependency` | dépendance à l'emprunt |
| `Research and development expense rate` | `research_and_development_expense_rate` | effort de R&D |
| `Quick Assets/Current Liability` | `quick_assets_current_liability` | couverture des dettes court terme |

**Méthode de sélection** (`src/data/select_features.py`, sur le jeu d'entraînement uniquement) :
1. Importance par permutation d'une Random Forest, en validation croisée stratifiée (5 plis × 5 répétitions), sur la PR-AUC.
2. Conservation des variables stables (importance moyenne − écart-type > 0).
3. Suppression des variables redondantes (|r| > 0,85). Exemple : ROA(C), corrélée à 0,99 avec ROA(B).

## 4. Protocole d'entraînement et de sélection

| Candidat | Recherche | Gestion du déséquilibre | Meilleurs paramètres |
|---|---|---|---|
| Régression logistique | GridSearchCV (`C`, `solver`) | `class_weight="balanced"` | `C=0.01`, `solver="lbfgs"` |
| Arbre de décision | GridSearchCV (profondeur, tailles minimales, critère) | `class_weight="balanced"` | `criterion="entropy"`, `max_depth=5`, `min_samples_leaf=5` |
| XGBoost | RandomizedSearchCV (25 tirages) | `scale_pos_weight ≈ 30` (sains / faillites) | `n_estimators=200`, `max_depth=3`, `learning_rate=0.01`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3` |

- **Validation croisée** : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ; métriques suivies : PR-AUC, Recall, Precision, F1, ROC-AUC ; `refit` sur la PR-AUC.
- **Règle** : le choix se fait sur la PR-AUC moyenne des 5 plis. Le jeu de test n'est ouvert qu'une fois, pour le modèle retenu.
- Les meilleurs paramètres ci-dessus proviennent d'un entraînement de référence. Ceux du champion en production figurent dans son run MLflow, qui fait foi.

## 5. Performances

### Validation croisée (jeu d'entraînement)

| Modèle | PR-AUC | Recall |
|---|---|---|
| Régression logistique | 0,343 (± 0,04) | 0,84 |
| Arbre de décision | 0,278 (± 0,05) | 0,76 |
| **XGBoost** | **0,415 (± 0,08)** | **0,83** |

PR-AUC d'un modèle aléatoire : 0,032. Pli par pli, XGBoost dépasse la régression logistique sur 4 plis sur 5.

### Test final (XGBoost, seuil 0,5)

| PR-AUC | Recall | Précision | Faillites détectées |
|---|---|---|---|
| 0,366 | 0,82 | 0,17 | 36 sur 44 |

**Lecture métier** : sur 1 364 entreprises, le modèle en signale environ 210 (15 % du portefeuille), qui contiennent 82 % des faillites. L'écart entre la validation croisée (0,415) et le test (0,366) est attendu : il reflète l'optimisme de la recherche d'hyperparamètres et la forte variance due au faible nombre de faillites.

> Le fichier `reports/training_results.json` provient d'une ancienne version du pipeline (94 variables, sans MLflow). Il ne décrit pas le modèle actuel.

## 6. Interprétation du score

- Le score **n'est pas calibré** : la pondération des faillites le pousse vers le haut. Il classe correctement les entreprises, mais un score de 0,80 ne signifie pas « 80 % de chances de faillite ».
- Le seuil de 0,5 est une valeur par défaut. En usage réel, on le fixerait avec le métier selon la capacité d'analyse. Exemple : « l'équipe peut examiner 100 dossiers par mois ».

## 7. Limites et risques

- **Représentativité** : entreprises taïwanaises cotées d'une période ancienne ; aucune dimension temporelle exploitable.
- **Faible nombre de faillites** : les estimations de performance ont une forte variance.
- **Sélection des variables** : faite sur tout le jeu d'entraînement avant la validation croisée. Les scores de validation croisée peuvent en être légèrement optimistes (une validation croisée imbriquée corrigerait ce biais).
- **Explicabilité** : pas encore d'explication locale (SHAP). Les contributions de variables ne doivent pas être lues comme des causes.
- **Équité et conformité** : non évaluées. Un usage réel exigerait une validation indépendante et le respect du cadre réglementaire applicable aux modèles de crédit.

## 8. Surveillance

- **Data drift** : chaque lot de production est comparé au jeu d'entraînement (test de Kolmogorov-Smirnov par variable ; alerte si au moins 50 % des variables dérivent). Voir [monitoring.md](monitoring.md).
- **Performance drift** : prévu. Il faut journaliser les prédictions, puis recalculer PR-AUC et Recall quand les issues réelles sont connues.
- **Réentraînement** : déclenché par un drift. Le nouveau modèle est enregistré en `@challenger` et ne remplace le champion qu'après décision humaine. Voir [runbook.md](runbook.md).
