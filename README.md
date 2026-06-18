# ChurnPredict — Système de Prédiction du Customer Churn

> Projet de Fin d'Année (PFA) — EMSI Rabat  
> 4ème année Ingénierie IA & Science des Données — Groupe 4IASDR6  
> Année universitaire 2025/2026

---

## Description

ChurnPredict est un système complet de prédiction du départ des clients (Customer Churn) dans le secteur des télécommunications. Il couvre l'ensemble de la chaîne de valeur du Machine Learning : exploration des données, prétraitement, modélisation, évaluation et déploiement via une application web interactive.

Le système compare trois algorithmes d'ensemble (Random Forest, XGBoost, CatBoost), optimise les hyperparamètres par GridSearchCV et expose les prédictions via une API REST Flask consommée par une interface React.js.

---

## Dataset

| Attribut | Valeur |
|---|---|
| Source | [Telco Customer Churn — IBM/Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Nombre de clients | 7 043 (7 032 après nettoyage) |
| Nombre de variables | 21 (20 features + 1 cible) |
| Variable cible | Churn (Yes = 1 / No = 0) |
| Clients churners | 1 869 (26.5%) |
| Clients fidèles | 5 163 (73.5%) |

### Variables principales

| Catégorie | Variables |
|---|---|
| Démographie | gender, SeniorCitizen, Partner, Dependents |
| Abonnement | tenure, Contract, PaperlessBilling |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV... |
| Facturation | MonthlyCharges, TotalCharges, PaymentMethod |
| Cible | Churn (Yes / No) |

---

## Pipeline Machine Learning

```
Dataset CSV
    ↓
Prétraitement
  ├── Suppression customerID
  ├── Conversion TotalCharges en numérique
  ├── Suppression 11 lignes manquantes
  ├── Encodage Churn (Yes=1 / No=0)
  ├── One-Hot Encoding (get_dummies) → 20 cols → 31 cols
  ├── Train/Test Split (80% / 20%, stratifié)
  └── Normalisation (StandardScaler)
    ↓
Modélisation
  ├── Random Forest (n=100)
  ├── GridSearchCV → RF Optimisé (n=200, depth=7)
  ├── XGBoost (n=100)
  └── CatBoost (iterations=100)
    ↓
Évaluation
  ├── Classification Report (Précision, Rappel, F1)
  ├── AUC-ROC
  ├── Matrices de confusion
  ├── Courbe ROC comparative
  └── Feature Importance (3 modèles)
    ↓
Sauvegarde
  ├── rf_model.pkl
  ├── xgb_model.pkl
  ├── cat_model.pkl
  ├── scaler.pkl
  └── columns.pkl
```

---

## Résultats des Modèles

| Modèle | Accuracy | AUC-ROC | Précision (Churn) | Rappel (Churn) | F1 (Churn) | Vrais Churns |
|---|---|---|---|---|---|---|
| **RF Optimisé** | **0.79** | **0.839** | 0.64 | 0.47 | 0.54 | 174 / 374 |
| CatBoost | 0.78 | 0.825 | 0.61 | 0.51 | 0.56 | 192 / 374 |
| XGBoost | 0.78 | 0.820 | 0.59 | **0.55** | **0.57** | **205 / 374** |

### Meilleurs hyperparamètres (GridSearchCV)
```
n_estimators    = 200
max_depth       = 7
min_samples_split = 2
AUC-ROC (CV)   = 0.8461
```

### Feature Importance — Variables clés communes
- **tenure** (ancienneté) → variable la plus prédictive (RF + CatBoost)
- **MonthlyCharges / TotalCharges** → impact financier fort
- **InternetService_Fiber optic** → risque élevé de churn
- **Contract_Two year** → fort indicateur de fidélité

---

## Architecture de l'Application

```
Utilisateur (Navigateur)
        ↕ HTTP
Frontend React.js (Port 3000)
  ├── Dashboard.js      → Statistiques + graphiques
  ├── PredictionForm.js → Prédiction individuelle
  ├── CompareForm.js    → Comparaison 3 modèles
  └── History.js        → Historique des prédictions
        ↕ Axios / JSON
Backend Flask (Port 5000)
  └── /predict [POST]   → Chargement pkl + prédiction
        ↕ Pickle
Modèles ML
  ├── rf_model.pkl
  ├── xgb_model.pkl
  └── cat_model.pkl
```

---

## Fonctionnalités

### 📊 Dashboard
- 4 cartes statistiques : total clients, clients perdus, taux de churn, meilleur AUC-ROC
- Graphique Doughnut : distribution fidèles / churners
- Graphique à barres : churn par type de contrat
- Graphique comparatif : performances des 3 modèles (Accuracy, AUC-ROC, F1-Score)

### 📋 Prédiction Individuelle
- Sélection du modèle : Random Forest / XGBoost / CatBoost
- Formulaire complet en 3 sections : Informations personnelles, Services, Facturation
- Résultat avec probabilité de churn et barre de progression colorée
- Recommandations de fidélisation personnalisées si client à risque

### ⚖️ Comparaison des 3 Modèles
- Soumission simultanée aux 3 modèles
- Affichage côte à côte des probabilités (RF / XGB / CatBoost)
- Verdict par vote majoritaire (≥ 2/3 modèles)
- Probabilité moyenne + détection des divergences entre modèles

### 🕓 Historique des Prédictions
- Tableau de suivi de toutes les prédictions de la session
- Colonnes : modèle, contrat, ancienneté, charges, probabilité, résultat, heure
- Filtres : Tous / À risque / Fidèles
- Barres de progression colorées (rouge = churn, vert = fidèle)

### 💡 Recommandations de Fidélisation

| Condition | Recommandation |
|---|---|
| Contrat mensuel | Proposer un contrat annuel avec réduction |
| Charges > 65$/mois | Offrir une réduction tarifaire |
| Ancienneté < 12 mois | Programme de fidélité nouveaux clients |
| Pas de support technique | Support technique gratuit 3 mois |
| Pas de sécurité en ligne | Sécurité en ligne offerte |
| Fiber optic | Audit qualité du service |
| Chèque électronique | Encourager le prélèvement automatique |

---

## Structure du Projet

```
churn-app/
├── backend/
│   ├── app.py              # API Flask (endpoint /predict)
│   ├── rf_model.pkl        # Random Forest optimisé (GridSearchCV)
│   ├── xgb_model.pkl       # XGBoost
│   ├── cat_model.pkl       # CatBoost
│   ├── scaler.pkl          # StandardScaler entraîné
│   ├── columns.pkl         # Ordre des 30 colonnes encodées
│   └── requirements.txt    # Dépendances Python
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.js           # Composant principal + routage
│       ├── App.css          # Styles globaux (sidebar, cards, forms)
│       └── components/
│           ├── Sidebar.js       # Navigation latérale (4 onglets)
│           ├── Dashboard.js     # Dashboard + Chart.js
│           ├── PredictionForm.js # Prédiction individuelle + recommandations
│           ├── CompareForm.js   # Comparaison 3 modèles simultanée
│           └── History.js       # Historique avec filtres
│
└── PFA.ipynb                # Notebook Google Colab (ML complet)
│
└── README.md
```

---

## Installation et Lancement

### Prérequis
- Python 3.10+
- Node.js 18+
- npm 9+

### Backend Flask

```bash
cd backend
pip install -r requirements.txt
python app.py
# → Serveur démarré sur http://127.0.0.1:5000
```

### Frontend React

```bash
cd frontend
npm install
npm start
# → Application accessible sur http://localhost:3000
```

---

## Technologies

### Machine Learning
| Outil | Usage |
|---|---|
| Python 3.12 | Langage principal |
| Google Colab | Environnement d'entraînement |
| Pandas / NumPy | Manipulation des données |
| Matplotlib / Seaborn | Visualisations EDA |
| Scikit-learn 1.9 | Pipeline ML, métriques, GridSearchCV |
| XGBoost 3.x | Gradient Boosting optimisé |
| CatBoost 1.2 | Boosting pour variables catégorielles |
| Pickle | Sérialisation des modèles |

### Application Web
| Outil | Usage |
|---|---|
| Flask | API REST backend |
| Flask-CORS | Cross-Origin Resource Sharing |
| React.js | Interface utilisateur frontend |
| Chart.js / React-Chartjs-2 | Graphiques interactifs |
| Axios | Requêtes HTTP depuis React |

---

## Auteurs

- **AZZAOUI Alae** — EMSI Rabat

## Encadrant

**M. ELGERARI Oussama** — EMSI Rabat

---

*EMSI Rabat — École Marocaine des Sciences de l'Ingénieur — Honoris United Universities*  
*Année Universitaire 2025/2026*
