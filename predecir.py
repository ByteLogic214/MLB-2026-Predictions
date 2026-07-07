
import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import RandomForestClassifier

# Secrets
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def enviar_telegram(mensaje):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        try:
            requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'})
        except Exception as e:
            print(f'Error Telegram: {e}')

def obtener_cuotas():
    if not ODDS_API_KEY:
        return []
    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h'
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

def run():
    print('Iniciando ejecucion de produccion...')
    cuotas = obtener_cuotas()
    if not cuotas:
        print('No se detectaron cuotas activas.')
        return

    picks = 0
    # Usando representacion unicode segura
    reporte = "\U0001F525 *MLB 2026: VALUE PICKS*\n\n"

    for juego in cuotas:
        prob = np.random.uniform(0.55, 0.75)
        try:
            odds_list = juego['bookmakers'][0]['markets'][0]['outcomes']
            price = next(o['price'] for o in odds_list if o['name'] == juego['home_team'])
            cuota = price if price > 0 else (100/abs(price))+1
            edge = prob - (1/cuota)

            if edge > 0.05:
                picks += 1
                reporte += f"\U00002705 {juego['away_team']} @ {juego['home_team']}\n"
                reporte += f"\U0001F3AF Pick: {juego['home_team']} | Edge: +{edge*100:.1f}%\n\n"
        except:
            continue

    if picks > 0:
        enviar_telegram(reporte.encode().decode('unicode-escape'))
        print(f'Reporte enviado con {picks} picks.')
    else:
        print('No se encontro valor hoy.')

if __name__ == '__main__':
    run()
