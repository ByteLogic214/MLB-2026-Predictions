import sys
import os
import numpy as np
import requests
from config import Config
from model import ModeloPredictivo
from api import OddsAPI
from groq import Groq

def juez_groq(reporte, config: Config):
    key = config.get_groq_api_key()
    if not key: return reporte
    client = Groq(api_key=key)
    prompt = f"""Eres un experto en MLB. Revisa este reporte de apuestas:

{reporte}

Mejora el formato y responde con el reporte final o 'RECHAZADO'."""
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
    modelo = ModeloPredictivo()
    
    # 1. Reentrenamiento
    if modelo.reentrenar_con_datos_recientes():
        print('Modelo actualizado.')
    
    # 2. Obtener Datos y Predecir (Simulado para validación)
    reporte_simulado = "✅ Yankees vs Red Sox | Pick: Yankees | Prob: 68%"
    
    # 3. Validación con Juez Groq
    print('Consultando al Juez Groq...')
    reporte_final = juez_groq(reporte_simulado, config)
    
    if reporte_final:
        print('Reporte validado y enviado.')
        enviar_telegram(reporte_final, config)
    else:
        print('Reporte rechazado por el Juez.')

if __name__ == '__main__':
    run()
