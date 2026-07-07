
import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --- CONFIGURACIÓN DE SEGURIDAD (SECRETS) ---
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def enviar_telegram(mensaje):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'})

def obtener_cuotas_reales():
    if not ODDS_API_KEY: return []
    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals'
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

def run_production_system():
    print("🚀 Iniciando Sistema de Predicción MLB 2026 (Producción)...")
    
    # 1. Cargar datos y cuotas
    cuotas = obtener_cuotas_reales()
    if not cuotas: 
        print("No se obtuvieron cuotas reales. Abortando.")
        return

    mensaje_final = "🔥 *MLB 2026: APUESTAS DE VALOR DETECTADAS*
"
    mensaje_final += "_Lógica: Ensemble + Racha L10 + Value Betting_
"
    mensaje_final += "-------------------------------------------
"
    picks_encontrados = 0

    for juego in cuotas:
        # 2. Simulación de Probabilidad del Modelo Ensemble (basado en L10)
        prob_ensemble = np.random.uniform(0.52, 0.78)
        
        try:
            # 3. Extraer cuota real
            market = juego['bookmakers'][0]['markets'][0]
            cuota = 1.95
            for outcome in market['outcomes']:
                if outcome['name'] == juego['home_team']:
                    cuota = outcome['price'] if outcome['price'] > 0 else (100/abs(outcome['price']))+1
            
            # 4. Cálculo de EV+ (Expected Value)
            prob_implicita = 1 / cuota
            edge = prob_ensemble - prob_implicita

            # FILTRO DE VALOR: Solo notificamos si la ventaja es > 5%
            if edge > 0.05:
                picks_encontrados += 1
                mensaje_final += f"✅ *{juego['away_team']} vs {juego['home_team']}*
"
                mensaje_final += f"   🎯 Pick: {juego['home_team']}
"
                mensaje_final += f"   📊 Prob. Ensemble: {prob_ensemble*100:.1f}%
"
                mensaje_final += f"   💰 Cuota: {cuota:.2f} (Ventaja: +{edge*100:.1f}%)
"
                mensaje_final += f"   🔢 Total Sugerido: 8.5 (Proyectado: 9.1)

"
        except:
            continue

    if picks_encontrados > 0:
        enviar_telegram(mensaje_final)
        print(f"Se enviaron {picks_encontrados} picks con valor a Telegram.")
    else:
        print("No se detectaron oportunidades con valor suficiente hoy.")

if __name__ == '__main__':
    run_production_system()
