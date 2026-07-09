import requests
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScraperESPN")

class ScraperESPN:
    def __init__(self):
        self.base_url = "https://www.espn.com/mlb/scoreboard"

    def get_odds(self, date_str: str) -> List[Dict]:
        logger.info(f"Consultando cuotas en ESPN para {date_str}...")
        return []
# Alias para compatibilidad con el resto del sistema
ESPNScraper = ScraperESPN
