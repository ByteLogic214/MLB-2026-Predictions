import os
import pandas as pd
import requests
from datetime import datetime
import logging
from scraper_espn import ESPNScraper  # Importamos tu scraper existente

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveDataFetcher:
    def __init__(self):
        self.raw_path = "data/raw"
        self.mlb_api_url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1"
        self.scraper_espn = ESPNScraper()
        os.makedirs(self.raw_path, exist_ok=True)

    def get_real_mlb_data(self, date_str="2026-07-08"):
        """
        Obtiene datos reales de la API de MLB y los cruza con ESPN.
        """
        logger.info(f"Extrayendo datos reales para la fecha: {date_str}")

        try:
            # 1. API Oficial de MLB
            response = requests.get(f"{self.mlb_api_url}&date={date_str}")
            games_data = response.json().get('dates', [])[0].get('games', [])

            mlb_list = []
            for game in games_data:
                mlb_list.append({
                    'game_id': game.get('gamePk'),
                    'home_team': game['teams']['home']['team']['name'],
                    'away_team': game['teams']['away']['team']['name'],
                    'status': game['status']['abstractGameState']
                })

            df_mlb = pd.DataFrame(mlb_list)

            # 2. Complementar con Scraper ESPN (Cuotas/Live)
            # El scraper se integrará aquí para capturar el mercado real en 2026

            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            final_path = os.path.join(self.raw_path, f"mlb_live_{timestamp}.csv")
            df_mlb.to_csv(final_path, index=False)

            logger.info(f"✅ {len(df_mlb)} juegos reales capturados en {final_path}")
            return df_mlb

        except Exception as e:
            logger.error(f"❌ Error capturando datos reales: {e}")
            return None

if __name__ == "__main__":
    fetcher = LiveDataFetcher()
    fetcher.get_real_mlb_data("2026-07-08")

# ── Aliases para compatibilidad con main.py ───────────────────────────────────
class LiveMLBDataFetcher(LiveDataFetcher):
    """Alias de LiveDataFetcher para compatibilidad con main.py."""
    pass


class LiveFeatureGenerator:
    """Generador de features en tiempo real (stub funcional)."""

    def __init__(self):
        pass

    def generate(self, game_data: dict) -> dict:
        """Genera features básicas desde datos en vivo."""
        return {
            'home_advantage': 0.035,
            'win_pct_diff_L5': 0.0,
            'run_diff_L5': 0.0,
            'momentum_home': 0.0,
            'momentum_away': 0.0,
        }
