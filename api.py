import requests
import logging
from config import Config

class OddsAPI:
    def __init__(self, config: Config):
        self.config = config
        self.base_url = 'https://api.the-odds-api.com/v4'

    def obtener_mercados_completos(self):
        if not self.config.get_odds_api_key():
            logging.error('API Key no encontrada.')
            return []
        
        url = f'{self.base_url}/sports/baseball_mlb/odds/'
        params = {
            'apiKey': self.config.get_odds_api_key(),
            'regions': 'us',
            'markets': 'h2h,totals,spreads', # Expandido a más mercados
            'oddsFormat': 'decimal'
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f'Error en API expandida: {e}')
            return []
