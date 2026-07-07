import requests
import logging
import time

logger = logging.getLogger(__name__)

# Máximo de reintentos para peticiones fallidas
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos entre reintentos


class OddsAPI:
    def __init__(self, config):
        self.config = config
        self.api_key = config.get_odds_api_key()
        self.base_url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/'

    def obtener_cuotas(self, sport='baseball_mlb'):
        """
        Obtiene las cuotas de MLB incluyendo h2h, totals y spreads.
        Incluye reintentos automáticos y manejo de rate limits.
        """
        if not self.api_key:
            logger.error('❌ No se encontró ODDS_API_KEY')
            return []

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,totals,spreads'
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f'📡 Consultando Odds API (intento {attempt}/{MAX_RETRIES})...'
                )
                response = requests.get(
                    self.base_url, params=params, timeout=15
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        f'✅ Cuotas obtenidas: {len(data)} eventos activos'
                    )
                    return data

                elif response.status_code == 429:
                    # Rate limit excedido - esperar más tiempo
                    retry_after = int(
                        response.headers.get('x-requests-remaining', 5)
                    )
                    logger.warning(
                        f'⏳ Rate limit alcanzado. '
                        f'Restantes: {response.headers.get("x-requests-remaining", "?")}. '
                        f'Esperando {RETRY_DELAY * attempt}s...'
                    )
                    time.sleep(RETRY_DELAY * attempt)
                    continue

                elif response.status_code == 401:
                    logger.error('🔑 API Key inválida o expirada.')
                    return []

                else:
                    logger.warning(
                        f'⚠️ Respuesta inesperada: HTTP {response.status_code}'
                    )
                    time.sleep(RETRY_DELAY * attempt)

            except requests.exceptions.Timeout:
                logger.warning(
                    f'⏰ Timeout en intento {attempt}/{MAX_RETRIES}'
                )
                time.sleep(RETRY_DELAY * attempt)

            except requests.exceptions.ConnectionError as e:
                logger.error(f'🔌 Error de conexión: {e}')
                time.sleep(RETRY_DELAY * attempt)

            except Exception as e:
                logger.error(f'❌ Error inesperado: {e}')
                return []

        logger.error('🚫 Todos los reintentos fallaron.')
        return []

    def remaining_requests(self):
        """Retorna el número de peticiones restantes (si disponible)."""
        if not self.api_key:
            return None
        try:
            url = f'{self.base_url}?apiKey={self.api_key}&regions=us&markets=h2h'
            response = requests.get(url, timeout=10)
            return response.headers.get('x-requests-remaining', None)
        except Exception:
            return None
