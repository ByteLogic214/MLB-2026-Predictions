import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelTraining")

def train_expert_models(data_path="datos_entrenamiento_mlb.csv"):
    if not os.path.exists(data_path):
        logger.error(f"No se encontró el archivo de datos: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # 1. Preprocesamiento simple para demostración (ajustar según columnas reales)
    # Seleccionamos numéricas y eliminamos objetivos/fechas
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'actual_winner' in features: features.remove('actual_winner')
    
    X = df[features].fillna(0)
    y = df['actual_winner']

    os.makedirs("models", exist_ok=True)

    # 2. Entrenar Random Forest
    logger.info("Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, "models/random_forest_v1.joblib")

    # 3. Entrenar XGBoost
    logger.info("Entrenando XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X, y)
    joblib.dump(xgb, "models/xgboost_v1.joblib")

    logger.info("✅ Modelos expertos guardados en la carpeta /models/")

if __name__ == '__main__':
    train_expert_models()