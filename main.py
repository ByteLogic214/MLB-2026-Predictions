import hashlib
import os
import logging
import pandas as pd
from config import Config
from api import OddsAPI
from model import MLBQuantModel

def run():
    logging.basicConfig(level=logging.INFO)
    print('🚀 MLB QUANT v4.0 - MULTI-MARKET & ANTI-DUPE')

    config = Config()
    api = OddsAPI(config)
    model = MLBQuantModel()
    
    log_file = 'sent_alerts.log'
    if not os.path.exists(log_file): 
        open(log_file, 'w').close()
    
    with open(log_file, 'r') as f: 
        hashes = set(f.read().splitlines())

    # Obtener cuotas para múltiples mercados
    eventos = api.obtener_cuotas(config.get_odds_api_key(), 'all')
    
    for ev in eventos:
        # Generar hash único para evitar duplicados
        unique_id = f"{ev.get('home_team')}_{ev.get('commence_time')}"
        event_hash = hashlib.md5(unique_id.encode()).hexdigest()
        
        if event_hash in hashes:
            continue

        print(f'📢 Analizando: {ev.get("away_team")} @ {ev.get("home_team")}')
        
        # Simulación de lógica de valor para diversos mercados
        # Aquí se integrarían las predicciones de model.py
        
        with open(log_file, 'a') as f:
            f.write(event_hash + '\n')

if __name__ == '__main__':
    run()
