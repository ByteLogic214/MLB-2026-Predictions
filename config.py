import os

class Config:
    def __init__(self):
        # Usar os.getenv para evitar bloqueos de seguridad y permitir configuracion en Actions
        self.ODDS_API_KEY = os.getenv('ODDS_API_KEY')
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    def get_odds_api_key(self): return self.ODDS_API_KEY
    def get_groq_api_key(self): return self.GROQ_API_KEY
    def get_telegram_token(self): return self.TELEGRAM_TOKEN
    def get_telegram_chat_id(self): return self.TELEGRAM_CHAT_ID
