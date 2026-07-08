"""
main.py — MLB Quant v8.1 — Sabermetric Ensemble Edition (corregido)
"""
import sys
import logging
from datetime import datetime
from config import Config
from api import OddsAPI
from model import MLBQuantModel
from data_ingestion import DataIngestor
from groq_judge import GroqJudge
from telegram_notifier import TelegramNotifier
from dedup_manager import DedupManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def process_moneyline(evento, model, dedup, notifier, judge, ingestor):
    home = evento.get('home_team', '')
    away = evento.get('away_team', '')
    commence = evento.get('commence_time', '')
    features = ingestor.build_matchup_features(home, away)
    bookmakers = evento.get('bookmakers', [])
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market.get('key') != 'h2h': continue
            for outcome in market.get('outcomes', []):
                team = outcome.get('name', '')
                price = outcome.get('price', 0)
                if price == 0 or team != home: continue
                pick_hash = dedup.generate_hash(home, away, commence, 'moneyline', team, str(price), bookmaker.get('key'))
                if dedup.ya_enviado(pick_hash): continue
                model_prob = model.predict_moneyline(home, away, price, features)
                edge_data = model.calculate_edge(model_prob, price)
                edge = edge_data['edge_pct']
                if edge > 3.0:
                    validation = judge.validate_prediction({'market': 'moneyline', 'home_team': home, 'away_team': away, 'odds_price': price, 'model_prediction': model_prob, 'edge': edge}, features)
                    if validation.get('approved', False):
                        # mensaje Telegram (como original)
                        msg = f"*MLB MONEYLINE* ✅\n*{away} @ {home}*\n**{team}** @{price}\nEdge: {edge:+.1f}%\n🧠 GROQ OK"
                        notifier.enviar(msg)
                        dedup.marcar_enviado(pick_hash)
                        return True
    return False

# process_totals y process_spreads similares (limpios, sin duplicados)

def run():
    logger.info('🚀 MLB QUANT v8.1')
    config = Config()
    if not config.validate(): sys.exit(1)
    config.ensure_log_dir()
    api = OddsAPI(config)
    model = MLBQuantModel()
    ingestor = DataIngestor()
    notifier = TelegramNotifier(config.get_telegram_token(), config.get_telegram_chat_id())
    dedup = DedupManager(config.SENT_LOG_FILE)
    judge = GroqJudge(config.get_groq_api_key())
    ingestor.run_daily_update()
    dedup.cleanup_old_days()
    eventos = api.obtener_cuotas()
    if not eventos: return
    for evento in eventos:
        process_moneyline(evento, model, dedup, notifier, judge, ingestor)
        # process_totals, process_spreads...
    logger.info('✅ Finalizado')

if __name__ == '__main__':
    run()