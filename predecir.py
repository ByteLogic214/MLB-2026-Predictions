
import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Secrets
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def obtener_cuotas_reales():
    if not ODDS_API_KEY:
        return []
    # Consulta a The Odds API (Mercado Moneyline, Region US)
    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h'
    response = requests.get(url)
    return response.json() if response.status_code == 200 else []

def enviar_telegram(mensaje):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'})

def run_value_pipeline():
    print("Buscando apuestas con valor real...")
    cuotas = obtener_cuotas_reales()
    
    mensaje = "🔥 *MLB 2026: APUESTAS CON VALOR (EV+)*
"
    mensaje += "_Solo se muestran picks con ventaja > 5%_
"
    mensaje += "-------------------------------------------
"
    encontradas = 0

    for juego in cuotas[:5]: # Ejemplo con los primeros 5 encontrados
        # 1. Probabilidad del Modelo Ensemble (Simulada para el ejemplo)
        prob_modelo = np.random.uniform(0.55, 0.75) 
        
        # 2. Obtener cuota de la casa de apuestas (ej. DraftKings)
        try:
            odds_data = juego['bookmakers'][0]['markets'][0]['outcomes']
            cuota_decimal = 1.95 # Valor por defecto
            for outcome in odds_data:
                if outcome['name'] == juego['home_team']:
                    cuota_decimal = outcome['price'] if outcome['price'] > 0 else (100/abs(outcome['price']))+1
            
            # 3. Calcular Probabilidad Implícita y Valor
            prob_implicita = 1 / cuota_decimal
            edge = prob_modelo - prob_implicita

            if edge > 0.05: # Solo si el valor es mayor al 5%
                encontradas += 1
                mensaje += f"✅ *{juego['away_team']} @ {juego['home_team']}*
"
                mensaje += f"   🎯 Pick: {juego['home_team']}
"
                mensaje += f"   📊 Prob. Modelo: {prob_modelo*100:.1f}%
"
                mensaje += f"   💰 Cuota: {cuota_decimal:.2f} (Implícita: {prob_implicita*100:.1f}%)
"
                mensaje += f"   📈 VENTAXA: +{edge*100:.1f}%

"
        except:
            continue

    if encontradas > 0:
        enviar_telegram(mensaje)
    else:
        print("No se encontraron apuestas con valor suficiente en este ciclo.")

if __name__ == '__main__':
    run_value_pipeline()
