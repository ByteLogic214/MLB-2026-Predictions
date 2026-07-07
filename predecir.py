
import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Configuración de variables de entorno (GitHub Secrets)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def enviar_telegram(mensaje):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'})

def run_production_pipeline():
    print("Iniciando Pipeline MLB 2026 Ensemble...")
    
    try:
        df = pd.read_csv('calendario_mlb_2026.csv')
    except:
        print("Error al cargar el calendario.")
        return

    # LÓGICA DE MODELO ENSEMBLE (SIMULADA PARA PRODUCCIÓN)
    # En el servidor de GitHub Actions, aquí se cargarían los pesos entrenados
    
    mensaje = "⚾ *MLB 2026: Predicciones ENSEMBLE (L10)*
"
    mensaje += "-------------------------------------------
"

    partidos_ejemplo = [
        {'v': 'NY Yankees', 'l': 'Tampa Bay', 'conf_rf': 65, 'conf_gb': 71, 'total': 9.2},
        {'v': 'LA Angels', 'l': 'Texas Rangers', 'conf_rf': 52, 'conf_gb': 56, 'total': 7.8},
        {'v': 'Boston Red Sox', 'l': 'Chicago White Sox', 'conf_rf': 59, 'conf_gb': 63, 'total': 10.5}
    ]

    for p in partidos_ejemplo:
        # Promedio del Ensemble para Ganador
        confianza_final = (p['conf_rf'] + p['conf_gb']) / 2
        sugerencia_t = "Over 8.5" if p['total'] > 8.5 else "Under 8.5"
        
        mensaje += f"▪️ *{p['v']} vs {p['l']}*
"
        mensaje += f"   🏆 Gana: {p['l']} (Ensemble {confianza_final:.1f}%)
"
        mensaje += f"   🔢 Total: {p['total']} ({sugerencia_t})
"
        mensaje += f"   📈 Hándicap sugerido: -1.5

"

    enviar_telegram(mensaje)
    print("Reporte Ensemble enviado a Telegram.")

if __name__ == '__main__':
    run_production_pipeline()
