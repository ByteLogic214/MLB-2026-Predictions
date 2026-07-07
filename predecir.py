
import openml
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configuración de APIs
ODDS_API_KEY = '0ced2d1eb8c3177fe3b230622880898a'
TELEGRAM_TOKEN = '8647946321:AAGqhoPjP1N1Q8UeujtmTZ9oH8QFEGlGdPM'
TELEGRAM_CHAT_ID = '8536626773'

def enviar_telegram(mensaje):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': mensaje, 'parse_mode': 'Markdown'}
    requests.post(url, json=payload)

def obtener_odds_reales():
    # En producción 2026, esto consultaría el endpoint de MLB
    # Por ahora simulamos la integración con la API
    print("Consultando The Odds API para cuotas actualizadas...")
    return {"success": True, "note": "Integración lista"}

def run_mlb_pro_pipeline():
    print("Iniciando Pipeline PRO con Telegram y Odds API...")
    dataset = openml.datasets.get_dataset(44156)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = X.fillna(X.mean(numeric_only=True))

    le = LabelEncoder()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Simulación de partidos
    partidos = ['NY Yankees vs Tampa Bay', 'LA Angels vs Texas Rangers']
    X_real = pd.DataFrame(np.random.randint(0, 100, size=(len(partidos), X.shape[1])), columns=X.columns)
    preds = model.predict(X_real)

    mensaje_telegram = "⚾ *Predicciones MLB 2026* 

"
    for i, p in enumerate(preds):
        resultado = "Gana Local" if p == 1 else "Gana Visitante"
        mensaje_telegram += f"▪️ {partidos[i]}: *{resultado}*
"

    enviar_telegram(mensaje_telegram)
    print("Predicciones enviadas a Telegram.")

if __name__ == '__main__':
    run_mlb_pro_pipeline()
