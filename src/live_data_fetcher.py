"""
═══════════════════════════════════════════════════════════════════════════
live_data_fetcher.py — Real-Time MLB Data Engine v3.0
═══════════════════════════════════════════════════════════════════════════
Sistema de datos en vivo con:
- MLB Official API (datos play-by-play)
- ESPN Live Scoreboard
- The Odds API (cuotas en tiempo real)
- WebSocket feeds (para updates cada 30s)
- Cache inteligente (evita spam de requests)
═══════════════════════════════════════════════════════════════════════════
"""

import requests
import statsapi
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from config import Config
from api import OddsAPI
import threading
import queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class LiveMLBDataFetcher:
    """
    Fetcher de datos MLB en tiempo real con múltiples fuentes.
    Optimizado para trading en vivo (latencia < 5s).
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.odds_api = OddsAPI(self.config)

        # Cache de datos en vivo (TTL: 30 segundos)
        self.live_cache = {}
        self.cache_ttl = 30  # segundos

        # Queue para updates asíncronos
        self.update_queue = queue.Queue()

        # Thread de actualización continua
        self.update_thread = None
        self.is_running = False

        # Session con retry
        self.session = self._create_session()

        logger.info('🔴 Live MLB Data Fetcher v3.0 inicializado')

    def _create_session(self) -> requests.Session:
        """Crea session HTTP optimizada."""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry_strategy = Retry(
            total=2,  # Solo 2 reintentos (velocidad)
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ═══════════════════════════════════════════════════════════════════
    # OBTENER JUEGOS EN VIVO (9 de Julio 2026)
    # ═══════════════════════════════════════════════════════════════════

    def get_live_games(self, date_str: str = '2026-07-09') -> List[Dict]:
        """
        Obtiene todos los juegos del día especificado con estado EN VIVO.
        
        Returns:
            List[Dict]: Lista de juegos con info en tiempo real
        """
        cache_key = f'live_games_{date_str}'
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        logger.info(f'🔴 Obteniendo juegos en vivo para {date_str}...')

        try:
            # MLB Official API
            schedule = statsapi.schedule(date=date_str)

            live_games = []
            for game in schedule:
                game_id = game['game_id']
                status = game['status']

                # Solo juegos en progreso
                if status not in ['In Progress', 'Live', 'Warmup', 'Delayed']:
                    continue

                # Obtener detalles del juego en vivo
                live_data = self._fetch_game_live_details(game_id)

                game_info = {
                    'game_id': game_id,
                    'date': date_str,
                    'status': status,
                    'away_team': game['away_name'],
                    'home_team': game['home_name'],
                    'away_score': game.get('away_score', 0),
                    'home_score': game.get('home_score', 0),
                    'inning': game.get('current_inning', 'N/A'),
                    'inning_state': game.get('inning_state', 'N/A'),
                    'venue': game.get('venue_name', 'N/A'),
                    'game_time': game.get('game_datetime'),
                    'summary': game.get('summary', ''),
                    **live_data  # Datos adicionales en vivo
                }

                live_games.append(game_info)

            self._save_to_cache(cache_key, live_games)
            logger.info(f'✅ {len(live_games)} juegos en vivo encontrados')
            return live_games

        except Exception as e:
            logger.error(f'❌ Error obteniendo juegos en vivo: {e}')
            return []

    def _fetch_game_live_details(self, game_id: int) -> Dict:
        """
        Obtiene detalles avanzados de un juego en vivo.
        Incluye: pitcher, batter, on-base, count, outs, etc.
        """
        try:
            # MLB API Feed
            live_feed = statsapi.get('game', {'gamePk': game_id})

            if not live_feed:
                return {}

            live_data = live_feed.get('liveData', {})
            plays = live_data.get('plays', {})
            linescore = live_data.get('linescore', {})

            return {
                'current_play': plays.get('currentPlay', {}).get('result', {}).get('description', 'N/A'),
                'balls': linescore.get('balls', 0),
                'strikes': linescore.get('strikes', 0),
                'outs': linescore.get('outs', 0),
                'inning_half': linescore.get('inningHalf', 'N/A'),
                'current_batter': plays.get('currentPlay', {}).get('matchup', {}).get('batter', {}).get('fullName', 'N/A'),
                'current_pitcher': plays.get('currentPlay', {}).get('matchup', {}).get('pitcher', {}).get('fullName', 'N/A'),
                'on_first': linescore.get('offense', {}).get('first', {}).get('fullName', None),
                'on_second': linescore.get('offense', {}).get('second', {}).get('fullName', None),
                'on_third': linescore.get('offense', {}).get('third', {}).get('fullName', None),
            }

        except Exception as e:
            logger.warning(f'⚠️ Error en detalles del juego {game_id}: {e}')
            return {}

    # ═══════════════════════════════════════════════════════════════════
    # CUOTAS EN VIVO (Live Odds)
    # ═══════════════════════════════════════════════════════════════════

    def get_live_odds(self, date_str: str = '2026-07-09') -> pd.DataFrame:
        """
        Obtiene cuotas EN VIVO desde The Odds API.
        Incluye movimiento de líneas en tiempo real.
        """
        cache_key = f'live_odds_{date_str}'
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        logger.info(f'💰 Obteniendo cuotas en vivo...')

        try:
            eventos = self.odds_api.obtener_cuotas()

            if not eventos:
                logger.warning('⚠️ No hay cuotas disponibles')
                return pd.DataFrame()

            odds_data = []
            for evento in eventos:
                # Filtrar por fecha
                commence_time = datetime.fromisoformat(evento['commence_time'].replace('Z', '+00:00'))
                if commence_time.date() != datetime.fromisoformat(date_str).date():
                    continue

                for bookmaker in evento.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')

                        for outcome in market.get('outcomes', []):
                            odds_data.append({
                                'timestamp': datetime.now().isoformat(),
                                'home_team': evento['home_team'],
                                'away_team': evento['away_team'],
                                'commence_time': evento['commence_time'],
                                'bookmaker': bookmaker['key'],
                                'market': market_key,
                                'team': outcome.get('name'),
                                'price': outcome.get('price'),
                                'point': outcome.get('point'),
                            })

            df = pd.DataFrame(odds_data)
            self._save_to_cache(cache_key, df.to_dict('records'))
            logger.info(f'✅ {len(df)} cuotas en vivo obtenidas')
            return df

        except Exception as e:
            logger.error(f'❌ Error obteniendo cuotas: {e}')
            return pd.DataFrame()

    def track_line_movement(self, game_id: int, market: str = 'h2h') -> pd.DataFrame:
        """
        Rastrea el movimiento de líneas para un juego específico.
        Útil para detectar sharp money.
        """
        cache_key = f'line_movement_{game_id}_{market}'
        history = self._get_from_cache(cache_key) or []

        # Agregar snapshot actual
        current_odds = self.get_live_odds()
        if not current_odds.empty:
            game_odds = current_odds[
                (current_odds['home_team'].str.contains(str(game_id), na=False)) |
                (current_odds['away_team'].str.contains(str(game_id), na=False))
            ]

            if not game_odds.empty:
                history.append({
                    'timestamp': datetime.now().isoformat(),
                    'odds': game_odds.to_dict('records')
                })

                self._save_to_cache(cache_key, history)

        return pd.DataFrame(history)

    # ═══════════════════════════════════════════════════════════════════
    # ESPN LIVE SCOREBOARD (Backup Source)
    # ═══════════════════════════════════════════════════════════════════

    def get_espn_live_scoreboard(self, date_str: str = '2026-07-09') -> List[Dict]:
        """
        Scraping del scoreboard en vivo de ESPN como fuente secundaria.
        """
        logger.info('📺 Consultando ESPN Live Scoreboard...')

        try:
            # ESPN usa formato YYYYMMDD
            date_formatted = date_str.replace('-', '')
            url = f'https://www.espn.com/mlb/scoreboard/_/date/{date_formatted}'

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Parsear scoreboard
            games = []
            game_cards = soup.find_all('section', class_='Scoreboard')

            for card in game_cards:
                try:
                    status_elem = card.find('div', class_='ScoreboardScoreCell__Time')
                    status = status_elem.get_text(strip=True) if status_elem else 'Unknown'

                    # Solo juegos en vivo
                    if 'Final' in status or 'Postponed' in status:
                        continue

                    teams = card.find_all('div', class_='ScoreCell__TeamName')
                    scores = card.find_all('div', class_='ScoreCell__Score')

                    if len(teams) >= 2 and len(scores) >= 2:
                        games.append({
                            'away_team': teams[0].get_text(strip=True),
                            'home_team': teams[1].get_text(strip=True),
                            'away_score': int(scores[0].get_text(strip=True) or 0),
                            'home_score': int(scores[1].get_text(strip=True) or 0),
                            'status': status,
                            'source': 'ESPN'
                        })

                except Exception as e:
                    logger.warning(f'⚠️ Error parseando game card: {e}')
                    continue

            logger.info(f'✅ {len(games)} juegos en vivo (ESPN)')
            return games

        except Exception as e:
            logger.error(f'❌ Error en ESPN Scoreboard: {e}')
            return []

    # ═══════════════════════════════════════════════════════════════════
    # ACTUALIZACIÓN CONTINUA (Background Thread)
    # ═══════════════════════════════════════════════════════════════════

    def start_live_updates(self, date_str: str = '2026-07-09', interval: int = 30):
        """
        Inicia thread de actualización continua cada X segundos.
        """
        if self.is_running:
            logger.warning('⚠️ Live updates ya están corriendo')
            return

        self.is_running = True

        def update_loop():
            logger.info(f'🔄 Iniciando updates continuos cada {interval}s...')

            while self.is_running:
                try:
                    # Actualizar juegos en vivo
                    live_games = self.get_live_games(date_str)
                    self.update_queue.put(('games', live_games))

                    # Actualizar cuotas
                    live_odds = self.get_live_odds(date_str)
                    self.update_queue.put(('odds', live_odds))

                    time.sleep(interval)

                except Exception as e:
                    logger.error(f'❌ Error en update loop: {e}')
                    time.sleep(interval)

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        logger.info('✅ Live updates thread iniciado')

    def stop_live_updates(self):
        """Detiene el thread de updates continuos."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info('🛑 Live updates detenidos')

    def get_latest_update(self, timeout: int = 1) -> Optional[Tuple[str, any]]:
        """Obtiene el último update del queue."""
        try:
            return self.update_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ═══════════════════════════════════════════════════════════════════
    # CACHE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _get_from_cache(self, key: str) -> Optional[any]:
        """Obtiene datos del cache si no han expirado."""
        if key not in self.live_cache:
            return None

        cached_data, timestamp = self.live_cache[key]
        if time.time() - timestamp > self.cache_ttl:
            del self.live_cache[key]
            return None

        return cached_data

    def _save_to_cache(self, key: str, data: any):
        """Guarda datos en cache con timestamp."""
        self.live_cache[key] = (data, time.time())

    def clear_cache(self):
        """Limpia todo el cache."""
        self.live_cache.clear()
        logger.info('🧹 Cache limpiado')


# ═══════════════════════════════════════════════════════════════════════
# INTEGRACIÓN CON MODELO ML (Features en Tiempo Real)
# ═══════════════════════════════════════════════════════════════════════

class LiveFeatureGenerator:
    """
    Genera features ML en tiempo real para predicciones in-game.
    """

    def __init__(self, fetcher: LiveMLBDataFetcher):
        self.fetcher = fetcher

    def generate_in_game_features(self, game_id: int) -> Dict:
        """
        Genera features para predicción in-game:
        - Score differential
        - Momentum indicators
        - Pitcher performance
        - Bullpen usage
        """
        try:
            # Obtener estado actual del juego
            live_data = self.fetcher._fetch_game_live_details(game_id)

            features = {
                'game_id': game_id,
                'current_inning': self._parse_inning(live_data.get('inning_half', 'Top 1st')),
                'score_diff': live_data.get('home_score', 0) - live_data.get('away_score', 0),
                'count_balls': live_data.get('balls', 0),
                'count_strikes': live_data.get('strikes', 0),
                'outs': live_data.get('outs', 0),
                'runners_on_base': self._count_runners(live_data),
                # Agregar más features según sea necesario
            }

            return features

        except Exception as e:
            logger.error(f'❌ Error generando features in-game: {e}')
            return {}

    def _parse_inning(self, inning_str: str) -> float:
        """Convierte inning a número decimal (ej: 'Top 3rd' -> 2.5)."""
        try:
            inning_num = int(''.join(filter(str.isdigit, inning_str)))
            is_bottom = 'Bot' in inning_str
            return inning_num + (0.5 if is_bottom else 0.0)
        except Exception:
            return 1.0

    def _count_runners(self, live_data: Dict) -> int:
        """Cuenta corredores en base."""
        count = 0
        if live_data.get('on_first'):
            count += 1
        if live_data.get('on_second'):
            count += 1
        if live_data.get('on_third'):
            count += 1
        return count


# ═══════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO (9 de Julio 2026)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Inicializar fetcher
    fetcher = LiveMLBDataFetcher()

    # 1. Obtener juegos en vivo del 9 de julio 2026
    target_date = '2026-07-09'
    live_games = fetcher.get_live_games(target_date)

    print(f'\n🔴 JUEGOS EN VIVO - {target_date}')
    print('=' * 80)
    for game in live_games:
        print(f"\n{game['away_team']} @ {game['home_team']}")
        print(f"Score: {game['away_score']} - {game['home_score']}")
        print(f"Inning: {game['inning']} ({game['inning_state']})")
        print(f"Status: {game['status']}")
        print(f"Current Play: {game.get('current_play', 'N/A')}")
        print(f"Count: {game.get('balls', 0)}-{game.get('strikes', 0)}, {game.get('outs', 0)} outs")

    # 2. Obtener cuotas en vivo
    live_odds = fetcher.get_live_odds(target_date)
    print(f'\n💰 CUOTAS EN VIVO')
    print('=' * 80)
    print(live_odds.head(10))

    # 3. Iniciar updates continuos (cada 30 segundos)
    fetcher.start_live_updates(target_date, interval=30)

    # Esperar algunos updates
    print('\n🔄 Esperando updates en tiempo real...')
    for i in range(5):
        time.sleep(31)
        update = fetcher.get_latest_update()
        if update:
            update_type, data = update
            print(f'\n✅ Update recibido: {update_type}')
            if update_type == 'games':
                print(f'   {len(data)} juegos actualizados')

    # Detener updates
    fetcher.stop_live_updates()

    # 4. Generar features en tiempo real
    if live_games:
        game_id = live_games[0]['game_id']
        feature_gen = LiveFeatureGenerator(fetcher)
        features = feature_gen.generate_in_game_features(game_id)
        print(f'\n🧠 FEATURES IN-GAME:')
        print(json.dumps(features, indent=2))
