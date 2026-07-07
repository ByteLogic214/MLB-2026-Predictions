
import pandas as pd
import numpy as np
import os
import requests
import traceback

# Obtener variables de entorno
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def enviar_telegram(mensaje):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        try:
            requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'})
        except Exception as e:
            print(f'Error enviando a Telegram: {e}')

def run():
    print('--- INICIANDO MODO DEBUG ---')
    
    if not ODDS_API_KEY:
        print('CRÍTICO: ODDS_API_KEY no encontrada en Secrets.')
        enviar_telegram('❌ *Error de Configuración*: Falta la clave de Odds API en GitHub Secrets.')
        return

    try:
        print('Consultando The Odds API...')
        url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h'
        res = requests.get(url)
        
        if res.status_code != 200:
            print(f'Error API: {res.status_code} - {res.text}')
            return
            
        cuotas = res.json()
        print(f'Juegos encontrados: {len(cuotas)}')

        if not cuotas:
            enviar_telegram('✅ *Sistema Online*: No hay juegos programados con cuotas ahora mismo.')
            return

        # Lógica de predicción mínima para verificar flujo
        reporte = '⚾ *MLB 2026: Monitor Activo*
'
        reporte += f'Se detectaron {len(cuotas)} juegos activos para análisis.
'
        enviar_telegram(reporte)
        print('Proceso completado con éxito.')

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f'FALLO EN EJECUCIÓN:
{error_trace}')
        enviar_telegram(f'⚠️ *Error en Producción*:
```{e}```')

if __name__ == "__main__":
    run()
