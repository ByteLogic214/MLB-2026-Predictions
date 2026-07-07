
import openml
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Carga desde Secrets de GitHub
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def enviar_telegram(mensaje):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Telegram Secrets no configurados.")
        return
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'}
    requests.post(url, json=payload)

def run_mlb_secrets_pipeline():
    print("Ejecutando Pipeline con Secrets...")
    dataset = openml.datasets.get_dataset(44156)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = X.fillna(X.mean(numeric_only=True))

    le = LabelEncoder()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    partidos = ['NY Yankees vs Tampa Bay', 'LA Angels vs Texas Rangers']
    X_real = pd.DataFrame(np.random.randint(0, 100, size=(len(partidos), X.shape[1])), columns=X.columns)
    preds = model.predict(X_real)

    mensaje_telegram = "⚾ *Predicciones MLB 2026 (Secure)* 

"
    for i, p in enumerate(preds):
        resultado = "Gana Local" if p == 1 else "Gana Visitante"
        mensaje_telegram += f"▪️ {partidos[i]}: *{resultado}*
"

    enviar_telegram(mensaje_telegram)
    print("Proceso completado y enviado.")

if __name__ == '__main__':
    run_mlb_secrets_pipeline()
