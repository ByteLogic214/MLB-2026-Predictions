import hashlib
import os
import logging
from config import Config
from api import OddsAPI
from model import MLBQuantModel

def run():
    logging.basicConfig(level=logging.INFO)
    print('🚀 MLB QUANT v5.0 - UNIFIED MARKETS (ML & TOTALS)')

    config = Config()
    api = OddsAPI(config)
    model = MLBQuantModel()
    
    log_file = 'sent_alerts.log'
    if not os.path.exists(log_file): 
        with open(log_file, 'w') as f: pass
    
    with open(log_file, 'r') as f: 
        sent_hashes = set(f.read().splitlines())

    # Obtenemos cuotas que incluyen h2h y totals
    eventos = api.obtener_cuotas()
    
    for ev in eventos:
        home = ev.get('home_team')
        away = ev.get('away_team')
        commence_time = ev.get('commence_time')
        
        # Procesamos ambos mercados: Moneyline (h2h) y Totales
        for market_type in ['h2h', 'totals']:
            # Hash único por evento y por tipo de mercado para evitar duplicados
            event_hash = hashlib.md5(f"{home}_{away}_{commence_time}_{market_type}".encode()).hexdigest()
            
            if event_hash in sent_hashes:
                continue

            # Lógica de predicción según el mercado
            prob_win, pred_total = model.predict_value(None)
            
            print(f'📊 Procesando {market_type}: {away} @ {home}')
            # En producción aquí se calcularía el EDGE y se enviaría a Telegram
            
            with open(log_file, 'a') as f:
                f.write(event_hash + '\n')
            sent_hashes.add(event_hash)

if __name__ == '__main__':
    run()