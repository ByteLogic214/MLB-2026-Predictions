import pandas as pd
import os
import logging
from sklearn.ensemble import RandomForestClassifier

class ModeloPredictivo:
    def __init__(self):
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.base_path = 'MLB-2026-Predictions/'

    def reentrenar(self):
        ruta = os.path.join(self.base_path, 'datos_entrenamiento_mlb.csv')
        if not os.path.exists(ruta):
            logging.error('Archivo de entrenamiento no encontrado.')
            return False
        try:
            df = pd.read_csv(ruta)
            # Selección inteligente de columnas numéricas
            X = df.select_dtypes(include=['number']).drop(columns=['class'], errors='ignore')
            y = df['class']
            self.modelo.fit(X, y)
            logging.info('Modelo reentrenado exitosamente.')
            return True
        except Exception as e:
            logging.error(f'Error en reentrenamiento: {e}')
            return False
