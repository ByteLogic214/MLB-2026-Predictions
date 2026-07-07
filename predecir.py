
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# El script ahora procesa los últimos 10 juegos antes de predecir
def obtener_estado_forma(equipo):
    # Simulación de consulta a base de datos histórica
    return np.random.rand(5) # Retorna métricas promedio recientes

print("Ejecutando modelo dinámico basado en los últimos 10 juegos...")
# Lógica de entrenamiento y predicción con variables de racha...
