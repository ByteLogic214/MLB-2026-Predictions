import logging
import pandas as pd
import numpy as np
from config import Config
from api import OddsAPI
from model import MLBQuantModel

def run():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('🚀 MLB QUANT PREDICTOR v3.0 - MULTI-MARKET')

    config = Config()
    api = OddsAPI(config)
    modelo = MLBQuantModel()

    # Entrenar con datos históricos reales
    master_data = 'MLB-2026-Predictions/datos_entrenamiento_mlb.csv'
    if os.path.exists(master_data):
        modelo.entrenar_con_contexto(master_data)

    eventos = api.obtener_mercados_completos() if hasattr(api, 'obtener_mercados_completos') else api.obtener_cuotas()
    print(f'Analizando {len(eventos)} partidos con contexto real...')

    for evento in eventos:
        try:
            home = evento.get('home_team', 'Unknown')
            away = evento.get('away_team', 'Unknown')

            # Features sintéticas para validación (en prod vienen de data_ingestion)
            features = pd.DataFrame([[3.5, 4.2, 1.05, 4.8]], 
                                    columns=['team_era', 'opp_era', 'venue_factor', 'avg_runs_last_10'])

            # Verificamos si el modelo tiene el método de predicción expandido
            if hasattr(modelo, 'predecir_todo'):
                prob_win, pred_total = modelo.predecir_todo(features)
                print(f'
⚾ {away} @ {home}')
                print(f'   - Predicción Carreras Totales: {pred_total:.2f}')
                print(f'   - Probabilidad Victoria Local: {prob_win*100:.1f}%')
            else:
                print(f'
⚾ {away} @ {home} - Analizado (Modo Básico)')
        except Exception as e:
            logging.error(f'Error procesando evento: {e}')
            continue

if __name__ == '__main__':
    run()
