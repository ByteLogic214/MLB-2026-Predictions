import numpy as np
import requests
import sys
import os
from groq import Groq

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from api import OddsAPI

def juez_groq(reporte, config: Config):
    key = config.get_groq_api_key()
    if not key: return reporte
    client = Groq(api_key=key)
    prompt = f"""Eres un experto en apuestas deportivas y analista de MLB. 
    Revisa el reporte:

    {reporte}

    Responde con el reporte mejorado o 'RECHAZADO'."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        res = completion.choices[0].message.content
        return None if 'RECHAZADO' in res.upper() else res
    except: return reporte

def enviar_telegram(mensaje, config: Config):
    token = config.get_telegram_token()
    chat_id = config.get_telegram_chat_id()
    if token and chat_id:
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        requests.post(url, json={'chat_id': chat_id, 'text': mensaje, 'parse_mode': 'Markdown'})

def run():
    print('--- INICIANDO SISTEMA PROFESIONAL ---')
    config = Config()
    reporte_simulado = "✅ Yankees vs Red Sox | Pick: Yankees | Prob: 68%"
    reporte_final = juez_groq(reporte_simulado, config)
    if reporte_final:
        print('Reporte validado y listo para envio.')
        enviar_telegram(reporte_final, config)

if __name__ == '__main__':
    run()
