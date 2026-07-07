import requests
import logging
from config import Config

class OddsAPI:
    def __init__(self, config: Config):
        self.config = config
        self.base_url = 'https://api.the-odds-api.com/v4'

    def obtener_cuotas(self):
        if not self.config.get_odds_api_key():
            logging.error('API Key de Odds no configurada.')
            return []
            
        url = f'{self.base_url}/sports/baseball_mlb/odds/'
        params = {
            'apiKey': self.config.get_odds_api_key(),
            'regions': 'us',
            'markets': 'h2h'
        }
        try:
            # Timeout estricto para evitar bloqueos en CI/CD
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f'Fallo crítico en OddsAPI: {e}')
            return []
