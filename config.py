import os
import logging

logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.ODDS_API_KEY = os.getenv('ODDS_API_KEY')
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        # Ruta del log de hashes (se reinicia diariamente)
        self.SENT_LOG_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'logs'
        )
        self.SENT_LOG_FILE = os.path.join(
            self.SENT_LOG_DIR, 'sent_alerts.log'
        )

    def get_odds_api_key(self):
        return self.ODDS_API_KEY

    def get_groq_api_key(self):
        return self.GROQ_API_KEY

    def get_telegram_token(self):
        return self.TELEGRAM_TOKEN

    def get_telegram_chat_id(self):
        return self.TELEGRAM_CHAT_ID

    def validate(self):
        missing = []
        if not self.ODDS_API_KEY:
            missing.append('ODDS_API_KEY')
        if not self.TELEGRAM_TOKEN:
            missing.append('TELEGRAM_BOT_TOKEN')
        if not self.TELEGRAM_CHAT_ID:
            missing.append('TELEGRAM_CHAT_ID')
        if missing:
            logger.warning(f'⚠️ Faltan variables de entorno: {missing}')
            return False
        return True

    def ensure_log_dir(self):
        """Crea el directorio de logs si no existe."""
        os.makedirs(self.SENT_LOG_DIR, exist_ok=True)
