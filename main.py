import logging
import pandas as pd
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
    modelo.entrenar_con_contexto('MLB-2026-Predictions/datos_entrenamiento_mlb.csv')

    eventos = api.obtener_mercados_completos()
    print(f'Analizando {len(eventos)} partidos con contexto real...')

    for evento in eventos:
        home = evento['home_team']
        away = evento['away_team']
        
        # Simulación de extracción de features para el ejemplo (en prod se obtienen de data_ingestion)
        features = pd.DataFrame([[3.5, 4.2, 1.05, 4.8]], columns=['team_era', 'opp_era', 'venue_factor', 'avg_runs_last_10'])
        
        prob_win, pred_total = modelo.predecir_todo(features)
        
        print(f'
⚾ {away} @ {home}')
        print(f'   - Predicción Carreras Totales: {pred_total:.2f}')
        print(f'   - Probabilidad Victoria Local: {prob_win*100:.1f}%')

if __name__ == '__main__':
    run()
