import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_data(filepath, labelpath):
    data = pd.read_csv(filepath)
    labels = pd.read_csv(labelpath)
    df = pd.merge(labels, data, on='Unnamed: 0').rename(columns={'Unnamed: 0': 'sample_id'})
    X = df.drop(columns=['sample_id', 'Class'])
    y = df['Class']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le

def build_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ))
    ])
    return pipeline

if __name__ == "__main__":
    X, y, encoder = load_data('data/TCGA-PANCAN-HiSeq-801x20531/data.csv', 'data/TCGA-PANCAN-HiSeq-801x20531/labels.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    sample_weights = np.array([weights[cls] for cls in y_train])

    # Lors du fit, on passe les poids

    mlflow.set_experiment("Cancer_Classification")
    with mlflow.start_run(run_name="XGBoost_Weighted"):
        model_pipeline = build_pipeline()
        model_pipeline.fit(X_train, y_train, clf__sample_weight=sample_weights)

        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=encoder.classes_, 
                    yticklabels=encoder.classes_, cmap='Blues')
        plt.title(f"Matrice de Confusion (Acc: {acc:.2%})")
        
        # Sauvegarde temporaire de l'image pour l'envoyer à MLflow
        temp_plot_path = "models/confusion_matrix.png"
        plt.savefig(temp_plot_path)
        mlflow.log_artifact(temp_plot_path) # Enregistre l'image dans l'interface
        plt.close()

        # --- SAUVEGARDE DU MODÈLE ---
        mlflow.sklearn.log_model(model_pipeline, "model_pipeline_v1")

        print(f"Run terminé avec succès. Accuracy: {acc:.4f}")

        print("--- Rapport de Classification ---")
        print(classification_report(y_test, y_pred, target_names=encoder.classes_))
        booster = model_pipeline.named_steps['clf']

        # 2. On crée un DataFrame avec les noms de gènes et leur importance
        # X.columns contient tes noms 'gene_0', 'gene_1', etc.
        feat_imp = pd.DataFrame({
            'gene': X.columns,
            'importance': booster.feature_importances_
        }).sort_values(by='importance', ascending=False)

        # 3. On enregistre le Top 20 en format CSV (Artefact MLflow)
        top_20_path = "models/top_20_genes_xgboost.csv"
        feat_imp.head(20).to_csv(top_20_path, index=False)
        mlflow.log_artifact(top_20_path)

        # 4. On crée et on log un graphique pour une lecture rapide
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='gene', data=feat_imp.head(20), palette='viridis')
        plt.title('Top 20 Genes - XGBoost Importance')
        plt.tight_layout()

        plot_path = "models/feature_importance_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        joblib.dump(model_pipeline, 'models/cancer_classifier_pipeline.joblib')
        joblib.dump(encoder, 'models/label_encoder.joblib')

        print("Modèles sauvegardés avec succès dans le dossier 'models/'")