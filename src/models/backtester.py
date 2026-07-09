import pandas as pd
import numpy as np
import logging
from src.models.ensemble_model import MLBEnsembleModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestingEngine")

class MLBBacktester:
    def __init__(self, model):
        self.model = model

    def run_test(self, historical_data_path):
        """
        Compara las predicciones del modelo contra resultados reales pasados.
        """
        logger.info(f"Cargando datos históricos desde {historical_data_path}...")
        
        try:
            df = pd.read_csv(historical_data_path)
            if 'actual_winner' not in df.columns:
                logger.error("Los datos históricos deben contener la columna 'actual_winner'.")
                return

            # Generar predicciones
            logger.info("Generando predicciones sobre el set histórico...")
            probabilities = self.model.predict(df)
            df['predicted_prob'] = probabilities
            df['prediction'] = (df['predicted_prob'] > 0.5).astype(int)

            # Calcular métricas
            correct = (df['prediction'] == df['actual_winner']).sum()
            accuracy = correct / len(df)
            
            logger.info(f"--- RESULTADOS DEL BACKTESTING ---")
            logger.info(f"Total de juegos analizados: {len(df)}")
            logger.info(f"Predicciones acertadas: {correct}")
            logger.info(f"Precisión Final (Accuracy): {accuracy:.2%}")
            
            return accuracy

        except Exception as e:
            logger.error(f"Error durante el backtesting: {e}")
            return None

if __name__ == "__main__":
    # Demo: Usar datos de entrenamiento para validar consistencia inicial
    model = MLBEnsembleModel()
    tester = MLBBacktester(model)
    # Nota: Se asume la existencia de datos históricos para la prueba
    tester.run_test("datos_entrenamiento_mlb.csv")