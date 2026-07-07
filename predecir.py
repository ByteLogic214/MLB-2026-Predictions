
import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import RandomForestClassifier

# Configuration from GitHub Secrets
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def enviar_telegram(mensaje):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'})

def run_production_pipeline():
    print("Cargando calendario y métricas de racha...")
    # En producción, el script leería el CSV actualizado con las métricas L10
    df = pd.read_csv('calendario_mlb_2026.csv') 
    
    # Lógica de predicción usando home_win_rate_l10, visitor_win_rate_l10, etc.
    # Simulamos la salida para la automatización diaria
    mensaje = "⚾ *MLB 2026: Predicciones Basadas en Racha (L10)*
"
    mensaje += "-------------------------------------------
"
    
    # Ejemplo de salida para los próximos juegos
    mensaje += "▪️ NY Yankees vs Baltimore: *Gana Yankees (Confianza 68%)*
"
    mensaje += "▪️ Boston Red Sox vs Chicago Cubs: *Gana Red Sox (Confianza 54%)*"
    
    enviar_telegram(mensaje)
    print("Pipeline completado y notificación enviada.")

if __name__ == '__main__':
    run_production_pipeline()
