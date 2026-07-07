import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class ModeloPredictivo:
    def __init__(self, model_path='modelo_mlb.joblib'):
        self.model_path = model_path
        self.modelo = RandomForestClassifier(n_estimators=100)

    def reentrenar_con_datos_recientes(self, csv_path='datos_entrenamiento_mlb.csv'):
        """Descarga/Carga datos y actualiza el modelo"""
        try:
            if not os.path.exists(csv_path): return False
            df = pd.read_csv(csv_path)
            X = df.drop('target', axis=1)
            y = df['target']
            self.modelo.fit(X, y)
            joblib.dump(self.modelo, self.model_path)
            return True
        except Exception as e:
            print(f'Error en reentrenamiento: {e}')
            return False

    def predecir(self, X):
        return self.modelo.predict(X)
