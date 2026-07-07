import requests
import logging

class OddsAPI:
    def __init__(self, config):
        self.config = config
        self.api_key = config.get_odds_api_key()

    def obtener_cuotas(self, api_key=None, sport='all'):
        key = api_key or self.api_key
        url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={key}&regions=us&markets=h2h,totals,spreads'
        try:
            response = requests.get(url, timeout=10)
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            logging.error(f'Error API: {e}')
            return []
