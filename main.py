import hashlib
import os
import logging
from config import Config
from api import OddsAPI
from model import MLBQuantModel

def run():
    logging.basicConfig(level=logging.INFO)
    print('🚀 MLB QUANT v4.5 - FULL MARKETS (ML & TOTALS)')

    config = Config()
    api = OddsAPI(config)
    model = MLBQuantModel()
    
    log_file = 'sent_alerts.log'
    if not os.path.exists(log_file): 
        with open(log_file, 'w') as f: pass
    
    with open(log_file, 'r') as f: 
        sent_hashes = set(f.read().splitlines())

    eventos = api.obtener_cuotas()
    
    for ev in eventos:
        home = ev.get('home_team')
        away = ev.get('away_team')
        commence_time = ev.get('commence_time')
        
        # Hash único por evento y mercado para evitar duplicados
        event_hash = hashlib.md5(f"{home}_{away}_{commence_time}".encode()).hexdigest()
        
        if event_hash in sent_hashes:
            continue

        # Simulación de Lógica de Valor
        prob_win, pred_total = model.predict_value(None) # En prod usaría features de ev
        
        print(f'📊 Procesando: {away} @ {home}')
        # Aquí se dispararía la lógica de Telegram (omitida para brevedad en test)
        
        with open(log_file, 'a') as f:
            f.write(event_hash + '\n')
        sent_hashes.add(event_hash)

if __name__ == '__main__':
    run()