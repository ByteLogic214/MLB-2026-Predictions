import requests
import logging

logger = logging.getLogger(__name__)

TELEGRAM_API = 'https://api.telegram.org'


class TelegramNotifier:
    """Módulo dedicado para enviar alertas a Telegram."""

    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        if not self.enabled:
            logger.warning(
                '⚠️ Telegram NO configurado. '
                'Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID'
            )

    def enviar(self, mensaje, parse_mode='Markdown'):
        """
        Envía un mensaje a Telegram.
        Retorna True si fue exitoso, False si falló.
        """
        if not self.enabled:
            logger.info(f'📱 [DEMO] Mensaje no enviado:\n{mensaje}')
            return False

        url = f'{TELEGRAM_API}/bot{self.token}/sendMessage'

        try:
            response = requests.post(
                url,
                json={
                    'chat_id': self.chat_id,
                    'text': mensaje,
                    'parse_mode': parse_mode,
                    'disable_web_page_preview': True
                },
                timeout=10
            )

            if response.status_code == 200:
                logger.info('✅ Mensaje enviado a Telegram exitosamente.')
                return True
            else:
                logger.error(
                    f'❌ Error Telegram HTTP {response.status_code}: '
                    f'{response.text}'
                )
                # Reintentar sin parse_mode por si hay error de formato
                if parse_mode:
                    return self.enviar(mensaje, parse_mode=None)
                return False

        except requests.exceptions.Timeout:
            logger.error('⏰ Timeout al enviar a Telegram.')
            return False
        except Exception as e:
            logger.error(f'❌ Error inesperado Telegram: {e}')
            return False
