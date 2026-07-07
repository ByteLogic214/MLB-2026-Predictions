import logging
import os
import sys
from datetime import datetime

from config import Config
from api import OddsAPI
from model import MLBQuantModel
from telegram_notifier import TelegramNotifier
from dedup_manager import DedupManager
from groq_judge import GroqJudge

# ============================================================
# Configuración de logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mlb_predictions.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('MLB-Predictions')


def calculate_implied_probability(odds_price):
    """Calcula la probabilidad implícita de una cuota decimal."""
    if odds_price > 0:
        return 100 / (odds_price + 100)
    else:
        return abs(odds_price) / (abs(odds_price) + 100)


def calculate_edge(true_prob, odds_price):
    """Calcula el EDGE: diferencia entre prob. real e implícita."""
    implied = calculate_implied_probability(odds_price)
    return (true_prob - implied) * 100


def process_moneyline(event, model, dedup, notifier, judge):
    """Procesa Moneyline con validación GROQ."""
    home = event.get('home_team', 'Home')
    away = event.get('away_team', 'Away')
    commence = event.get('commence_time', 'N/A')

    bookmakers = event.get('bookmakers', [])
    if not bookmakers:
        return False

    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') != 'h2h':
                continue

            outcomes = market.get('outcomes', [])
            for outcome in outcomes:
                if outcome.get('name') == home:
                    price = outcome.get('price', 0)
                    if price == 0:
                        continue

                    pick_hash = dedup.generate_hash(
                        home, away, commence, 'moneyline',
                        bookmaker.get('key', 'unknown')
                    )

                    if dedup.ya_enviado(pick_hash):
                        continue

                    # Predicción del modelo
                    true_prob = model.predict_moneyline(home, away, price)
                    edge = calculate_edge(true_prob, price)

                    logger.info(
                        f'📊 ML: {away} @ {home} | '
                        f'Cuota: {price} | Prob: {true_prob*100:.1f}% | '
                        f'Edge: {edge:+.1f}%'
                    )

                    if edge > 3.0:
                        # Validar con GROQ Judge
                        validation = judge.validate_prediction({
                            'market': 'moneyline',
                            'home_team': home,
                            'away_team': away,
                            'odds_price': price,
                            'model_prediction': true_prob,
                            'edge': edge
                        })

                        if validation.get('approved', False):
                            confidence = validation.get('confidence', 0.5)
                            reasoning = validation.get('reasoning', '')
                            
                            msg = (
                                f"*MLB 2026 - VALUE PICK* ✅\n\n"
                                f"*Moneyline*\n"
                                f"*{away}* @ *{home}*\n"
                                f"Bookmaker: {bookmaker.get('title', 'N/A')}\n"
                                f"Cuota: `{price}`\n"
                                f"Prob. modelo: `{true_prob*100:.1f}%`\n"
                                f"Prob. implícita: `{calculate_implied_probability(price)*100:.1f}%`\n"
                                f"**EDGE: +{edge:.1f}%**\n\n"
                                f"🧠 *GROQ Judge*: Aprobado\n"
                                f"Confianza: `{confidence*100:.0f}%`\n"
                                f"_{reasoning}_\n\n"
                                f"⏰ {commence}"
                            )
                            notifier.enviar(msg)
                            dedup.marcar_enviado(pick_hash)
                            return True
                        else:
                            logger.warning(
                                f'⚠️ GROQ rechazó ML pick: {validation.get("reasoning", "")}'
                            )

    return False


def process_totals(event, model, dedup, notifier, judge):
    """Procesa Totals con validación GROQ."""
    home = event.get('home_team', 'Home')
    away = event.get('away_team', 'Away')
    commence = event.get('commence_time', 'N/A')

    bookmakers = event.get('bookmakers', [])
    if not bookmakers:
        return False

    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') != 'totals':
                continue

            outcomes = market.get('outcomes', [])
            total_line = market.get('point', 0)

            for outcome in outcomes:
                side = outcome.get('name', '')
                price = outcome.get('price', 0)
                if price == 0 or side == '':
                    continue

                pick_hash = dedup.generate_hash(
                    home, away, commence, 'totals',
                    side, str(total_line),
                    bookmaker.get('key', 'unknown')
                )

                if dedup.ya_enviado(pick_hash):
                    continue

                predicted_total = model.predict_total(home, away)
                edge = 0.0

                if side == 'Over' and predicted_total > total_line:
                    edge = ((predicted_total - total_line) / total_line) * 10
                elif side == 'Under' and predicted_total < total_line:
                    edge = ((total_line - predicted_total) / total_line) * 10

                logger.info(
                    f'📊 Total: {away} @ {home} | {side} {total_line} | '
                    f'Pred: {predicted_total:.1f} | Edge: {edge:+.1f}%'
                )

                if edge > 1.5:
                    # Validar con GROQ
                    validation = judge.validate_prediction({
                        'market': 'totals',
                        'home_team': home,
                        'away_team': away,
                        'odds_price': price,
                        'model_prediction': predicted_total,
                        'edge': edge,
                        'total_line': total_line,
                        'side': side
                    })

                    if validation.get('approved', False):
                        confidence = validation.get('confidence', 0.5)
                        reasoning = validation.get('reasoning', '')
                        
                        msg = (
                            f"*MLB 2026 - TOTAL PICK* ✅\n\n"
                            f"*{away}* @ *{home}*\n"
                            f"Bookmaker: {bookmaker.get('title', 'N/A')}\n"
                            f"**{side} {total_line}** runs\n"
                            f"Cuota: `{price}`\n"
                            f"Predicción: `{predicted_total:.1f}`\n"
                            f"**EDGE: +{edge:.1f}%**\n\n"
                            f"🧠 *GROQ Judge*: Aprobado\n"
                            f"Confianza: `{confidence*100:.0f}%`\n"
                            f"_{reasoning}_\n\n"
                            f"⏰ {commence}"
                        )
                        notifier.enviar(msg)
                        dedup.marcar_enviado(pick_hash)
                        return True

    return False


def process_spreads(event, model, dedup, notifier, judge):
    """Procesa Handicaps (Spreads) con validación GROQ."""
    home = event.get('home_team', 'Home')
    away = event.get('away_team', 'Away')
    commence = event.get('commence_time', 'N/A')

    bookmakers = event.get('bookmakers', [])
    if not bookmakers:
        return False

    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') != 'spreads':
                continue

            outcomes = market.get('outcomes', [])
            
            for outcome in outcomes:
                team = outcome.get('name', '')
                spread_line = outcome.get('point', 0)
                price = outcome.get('price', 0)
                
                if price == 0:
                    continue

                pick_hash = dedup.generate_hash(
                    home, away, commence, 'spreads',
                    team, str(spread_line),
                    bookmaker.get('key', 'unknown')
                )

                if dedup.ya_enviado(pick_hash):
                    continue

                # Predicción del spread
                predicted_spread = model.predict_spread(home, away, team)
                edge = 0.0

                # Si el spread predicho supera la línea, hay valor
                if team == home:
                    edge = (predicted_spread - spread_line) * 5
                else:
                    edge = (predicted_spread - spread_line) * 5

                logger.info(
                    f'📊 Spread: {team} {spread_line:+.1f} | '
                    f'Pred: {predicted_spread:+.1f} | Edge: {edge:+.1f}%'
                )

                if abs(edge) > 2.0:
                    # Validar con GROQ
                    validation = judge.validate_prediction({
                        'market': 'spreads',
                        'home_team': home,
                        'away_team': away,
                        'odds_price': price,
                        'model_prediction': predicted_spread,
                        'edge': edge,
                        'spread_line': spread_line
                    })

                    if validation.get('approved', False):
                        confidence = validation.get('confidence', 0.5)
                        reasoning = validation.get('reasoning', '')
                        
                        msg = (
                            f"*MLB 2026 - SPREAD PICK* ✅\n\n"
                            f"*{away}* @ *{home}*\n"
                            f"Bookmaker: {bookmaker.get('title', 'N/A')}\n"
                            f"**{team} {spread_line:+.1f}**\n"
                            f"Cuota: `{price}`\n"
                            f"Predicción: `{predicted_spread:+.1f}`\n"
                            f"**EDGE: {edge:+.1f}%**\n\n"
                            f"🧠 *GROQ Judge*: Aprobado\n"
                            f"Confianza: `{confidence*100:.0f}%`\n"
                            f"_{reasoning}_\n\n"
                            f"⏰ {commence}"
                        )
                        notifier.enviar(msg)
                        dedup.marcar_enviado(pick_hash)
                        return True

    return False


def run():
    """Función principal."""
    logger.info('=' * 50)
    logger.info('🚀 MLB QUANT v7.0 - GROQ JUDGE EDITION')
    logger.info(f'📅 {datetime.utcnow().isoformat()}Z')
    logger.info('=' * 50)

    config = Config()

    if not config.validate():
        logger.error('❌ Configuración incompleta.')
        sys.exit(1)

    config.ensure_log_dir()

    api = OddsAPI(config)
    model = MLBQuantModel()
    notifier = TelegramNotifier(
        config.get_telegram_token(),
        config.get_telegram_chat_id()
    )
    dedup = DedupManager(config.SENT_LOG_FILE)
    judge = GroqJudge(config.get_groq_api_key())

    dedup.cleanup_old_days()

    eventos = api.obtener_cuotas()

    if not eventos:
        logger.info('📭 Sin eventos activos.')
        return

    logger.info(f'📊 Procesando {len(eventos)} eventos...')

    ml_picks = 0
    total_picks = 0
    spread_picks = 0
    errors = 0

    for evento in eventos:
        try:
            if process_moneyline(evento, model, dedup, notifier, judge):
                ml_picks += 1
        except Exception as e:
            errors += 1
            logger.error(f'❌ Error ML: {e}')

        try:
            if process_totals(evento, model, dedup, notifier, judge):
                total_picks += 1
        except Exception as e:
            errors += 1
            logger.error(f'❌ Error Totals: {e}')

        try:
            if process_spreads(evento, model, dedup, notifier, judge):
                spread_picks += 1
        except Exception as e:
            errors += 1
            logger.error(f'❌ Error Spreads: {e}')

    total_sent = ml_picks + total_picks + spread_picks
    logger.info('=' * 50)
    logger.info(
        f'📋 RESUMEN: ML={ml_picks} | Totals={total_picks} | '
        f'Spreads={spread_picks} | TOTAL={total_sent}'
    )
    if errors > 0:
        logger.warning(f'⚠️ {errors} errores')
    logger.info('=' * 50)

    if total_sent > 0:
        summary = (
            f"*MLB 2026 - Resumen Horario*\n\n"
            f"⏰ {datetime.utcnow().strftime('%H:%M UTC')}\n"
            f"📊 Eventos: {len(eventos)}\n"
            f"✅ Picks: {total_sent}\n"
            f"  - Moneyline: {ml_picks}\n"
            f"  - Totals: {total_picks}\n"
            f"  - Spreads: {spread_picks}\n\n"
            f"🧠 Validados por GROQ Judge"
        )
        notifier.enviar(summary)


if __name__ == '__main__':
    run()
