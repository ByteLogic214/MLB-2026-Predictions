import os
import logging

logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.ODDS_API_KEY = os.getenv('ODDS_API_KEY')
        self.GROQ_API_KEY = os.getenv('QROQ_API_KEY') or os.getenv('GROQ_API_KEY')
        self.TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        self.SENT_LOG_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'logs'
        )
        self.SENT_LOG_FILE = os.path.join(self.SENT_LOG_DIR, 'sent_alerts.log')

    def get_odds_api_key(self):
        return self.ODDS_API_KEY

    def get_groq_api_key(self):
        return self.GROQ_API_KEY

    def get_telegram_token(self):
        return self.TELEGRAM_TOKEN

    def get_telegram_chat_id(self):
        return self.TELEGRAM_CHAT_ID

    def validate(self):
        """
        Valida configuración mínima para operar.
        ODDS_API_KEY es opcional (modo scan sin cuotas de mercado).
        Solo falla si falta Telegram (necesario para alertas).
        """
        missing_critical = []
        warnings_only = []

        if not self.ODDS_API_KEY:
            warnings_only.append('ODDS_API_KEY (modo sin cuotas de mercado)')
        if not self.TELEGRAM_TOKEN:
            missing_critical.append('TELEGRAM_BOT_TOKEN')
        if not self.TELEGRAM_CHAT_ID:
            missing_critical.append('TELEGRAM_CHAT_ID')

        if warnings_only:
            logger.warning(f'⚠️ Opcionales no configurados: {warnings_only}')

        if missing_critical:
            logger.error(f'❌ Críticos faltantes: {missing_critical}')
            return False

        return True

    def ensure_log_dir(self):
        os.makedirs(self.SENT_LOG_DIR, exist_ok=True)
