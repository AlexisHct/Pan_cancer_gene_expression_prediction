from os import getcwd
import pandas as pd
import joblib
import numpy as np
import sys

def load_inferance_tools():
    try:
        model = joblib.load('models/cancer_classifier_pipeline.joblib')
        encoder = joblib.load('models/label_encoder.joblib')
        return model, encoder
    except FileNotFoundError as e:
        print(f"Erreur : Fichier de modèle introuvable. Training potentiellement manquant")
        sys.exit(1)

def run_prediction(input_data):
    pipeline, encoder = load_inferance_tools()

    predictions_idx = pipeline.predict(input_data)

    probabilities = pipeline.predict_proba(input_data)

    cancer_types = encoder.inverse_transform(predictions_idx)

    confidences = np.max(probabilities, axis = 1)

    return cancer_types, confidences


if __name__ == "__main__":

    data_path = 'data/TCGA-PANCAN-HiSeq-801x20531/data.csv'
    label_path = 'data/TCGA-PANCAN-HiSeq-801x20531/labels.csv'

    try :
        data = pd.read_csv(data_path)
        labels = pd.read_csv(label_path)

        df = pd.merge(labels, data, on='Unnamed: 0').drop(columns=['Unnamed: 0', 'Class'], errors='ignore')
        
        sample = df.iloc[[10]]

        label, confidence = run_prediction(sample)

        print(f"Verdict du modèle : {label[0]}")
        print(f"Indice de confiance : {confidence[0]:.2%}")

        if confidence[0] < 0.80:
            print("Note : La confiance est modérée. Une analyse complémentaire est recommandée.")

    except Exception as e:
        print(f"Une erreur est survenue lors du test : {e}")
