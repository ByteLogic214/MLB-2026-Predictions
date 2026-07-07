import requests
from config import Config

class OddsAPI:
    def __init__(self, config: Config):
        self.config = config
        self.base_url = 'https://api.the-odds-api.com/v4'

    def obtener_cuotas(self):
        key = self.config.get_odds_api_key()
        if not key: return []
        url = f'{self.base_url}/sports/baseball_mlb/odds/'
        params = {'apiKey': key, 'regions': 'us', 'markets': 'h2h'}
        try:
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else []
        except: return []
