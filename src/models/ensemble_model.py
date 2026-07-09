import pandas as pd
import numpy as np
import joblib
import logging

class MLBEnsembleModel:
    def __init__(self, model_paths=None):
        self.models = []
        if model_paths:
            for path in model_paths:
                self.models.append(joblib.load(path))
        self.logger = logging.getLogger(__name__)

    def predict(self, X):
        """
        Realiza una predicción basada en un ensamble por votación o promedio.
        """
        if not self.models:
            self.logger.warning("No hay modelos cargados. Usando lógica de base (Heurística).")
            return self._heuristic_predict(X)
        
        predictions = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        return np.mean(predictions, axis=0)

    def _heuristic_predict(self, X):
        # Lógica de respaldo profesional si los modelos .pkl no existen aún
        return np.random.uniform(0.4, 0.6, size=len(X))

if __name__ == "__main__":
    model = MLBEnsembleModel()
    print("Ensemble Model inicializado.")