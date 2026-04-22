# Pan-Cancer Gene Expression Classification
## Présentation du Projet

Ce projet implémente un pipeline de Machine Learning capable de classifier 5 types de tumeurs cancéreuses à partir de données d'expression génétique (RNA-Seq). Le jeu de données (TCGA Pan-Cancer) présente un défi majeur : 20 531 gènes pour seulement 801 échantillons, créant un risque élevé de sur-apprentissage (overfitting).

Types de cancers classifiés : BRCA (Sein), KIRC (Rein), COAD (Colon), LUAD (Poumon), PRAD (Prostate).

## Architecture Technique & Choix Stratégiques
### 1. Gestion de la Haute Dimensionnalité

Une analyse exploratoire par PCA a été menée pour valider la séparabilité des classes. Le modèle final utilise XGBoost au sein d'un pipeline automatisé, permettant de capturer des interactions non-linéaires complexes entre les gènes sans réduction de dimension préalable forcée.

### 2. Équilibrage des Classes (Class Weighting)

Pour contrer la prévalence du cancer BRCA (classe majoritaire), j'ai implémenté une stratégie de pondération dynamique des classes (compute_class_weight).

* Résultat : Réduction drastique des faux positifs dans les classes minoritaires (LUAD/COAD).

### 3. Tracking & Reproductibilité (MLOps)

L'intégralité des expériences est tracée avec MLflow. Chaque "run" enregistre :

* Les hyperparamètres (n_estimators, learning_rate, etc.).
* Les métriques de performance (Accuracy, F1-Score).
* Les artefacts (Matrice de confusion, Feature Importance).

## Performances

* Accuracy Globale : 99%+
* Confiance Moyenne : ~99.72% sur les nouveaux échantillons.
* Interprétabilité : Extraction des Top 20 gènes les plus prédictifs pour chaque type de tumeur, permettant une validation biologique potentielle des biomarqueurs identifiés.

##  Installation et Utilisation
### Installation

```bash
git clone https://github.com/AlexisHct/Pan_cancer_gene_expression_prediction.git
cd Pan_cancer_gene_expression_prediction
pip install -r requirements.txt
```

### Entraînement et Tracking

```bash
python src/training.py
mlflow ui  # Pour visualiser les résultats sur http://localhost:5000
```
### Inférence (Prédiction)

```bash
python src/predict.py
```

## Structure du Dépôt
 
* src/training.py : Pipeline d'entraînement, gestion des poids et logging MLflow.
* src/predict.py : Script d'inférence autonome pour le diagnostic de nouveaux échantillons.
* models/ : Modèles sérialisés (.joblib) pour une utilisation en production.
* requirements.txt : Liste des dépendances figées pour la reproductibilité.