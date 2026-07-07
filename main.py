import logging
import os
import sys
from datetime import datetime

from config import Config
from api import OddsAPI
from model import MLBQuantModel
from telegram_notifier import TelegramNotifier
from dedup_manager import DedupManager

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
    """
    Calcula el EDGE: diferencia entre probabilidad real estimada
    y la probabilidad implícita de la cuota.
    Retorna el edge como porcentaje.
    """
    implied = calculate_implied_probability(odds_price)
    return (true_prob - implied) * 100


def process_moneyline(event, model, dedup, notifier):
    """
    Procesa el mercado Moneyline (h2h) de un evento.
    Retorna True si se envió un pick.
    """
    home = event.get('home_team', 'Home')
    away = event.get('away_team', 'Away')
    commence = event.get('commence_time', 'N/A')

    # Buscar cuota del home team en bookmakers
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

                    # Generar hash único para este pick
                    pick_hash = dedup.generate_hash(
                        home, away, commence, 'moneyline',
                        bookmaker.get('key', 'unknown')
                    )

                    if dedup.ya_enviado(pick_hash):
                        logger.debug(
                            f'⏭️ ML duplicado ignorado: {away} @ {home}'
                        )
                        continue

                    # Obtener predicción del modelo
                    true_prob = model.predict_moneyline(
                        home, away, price
                    )
                    edge = calculate_edge(true_prob, price)

                    logger.info(
                        f'📊 ML: {away} @ {home} | '
                        f'Cuota: {price} | '
                        f'Prob real: {true_prob*100:.1f}% | '
                        f'Edge: {edge:+.1f}%'
                    )

                    # Enviar si hay valor suficiente (>3% edge)
                    if edge > 3.0:
                        msg = (
                            f"*MLB 2026 - VALUE PICK*\n\n"
                            f"*Moneyline*\n"
                            f"*{away}* @ *{home}*\n"
                            f"Bookmaker: {bookmaker.get('title', 'N/A')}\n"
                            f"Cuota: `{price}`\n"
                            f"Prob. estimada: `{true_prob*100:.1f}%`\n"
                            f"Prob. implícita: "
                            f"`{calculate_implied_probability(price)*100:.1f}%`\n"
                            f"**EDGE: +{edge:.1f}%**\n\n"
                            f"⏰ {commence}"
                        )
                        notifier.enviar(msg)
                        dedup.marcar_enviado(pick_hash)
                        return True

    return False


def process_totals(event, model, dedup, notifier):
    """
    Procesa el mercado de Totals (Over/Under) de un evento.
    Retorna True si se envió un pick.
    """
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
                side = outcome.get('name', '')  # 'Over' o 'Under'
                price = outcome.get('price', 0)
                if price == 0 or side == '':
                    continue

                # Hash único por evento + mercado + lado + bookmaker
                pick_hash = dedup.generate_hash(
                    home, away, commence, 'totals',
                    side, str(total_line),
                    bookmaker.get('key', 'unknown')
                )

                if dedup.ya_enviado(pick_hash):
                    logger.debug(
                        f'⏭️ Total duplicado ignorado: '
                        f'{away} @ {home} {side} {total_line}'
                    )
                    continue

                # Obtener predicción del modelo para totals
                predicted_total = model.predict_total(home, away)
                edge = 0.0

                if side == 'Over' and predicted_total > total_line:
                    edge = ((predicted_total - total_line) / total_line) * 10
                elif side == 'Under' and predicted_total < total_line:
                    edge = ((total_line - predicted_total) / total_line) * 10

                logger.info(
                    f'📊 Total: {away} @ {home} | '
                    f'{side} {total_line} | Cuota: {price} | '
                    f'Predicción: {predicted_total:.1f} | Edge: {edge:+.1f}%'
                )

                if edge > 1.5:
                    msg = (
                        f"*MLB 2026 - TOTAL PICK*\n\n"
                        f"*{away}* @ *{home}*\n"
                        f"Bookmaker: {bookmaker.get('title', 'N/A')}\n"
                        f"**{side} {total_line}** runs\n"
                        f"Cuota: `{price}`\n"
                        f"Predicción modelo: `{predicted_total:.1f}`\n"
                        f"**EDGE: +{edge:.1f}%**\n\n"
                        f"⏰ {commence}"
                    )
                    notifier.enviar(msg)
                    dedup.marcar_enviado(pick_hash)
                    return True

    return False


def run():
    """Función principal de ejecución."""
    logger.info('=' * 50)
    logger.info('🚀 MLB QUANT v6.0 - SISTEMA UNIFICADO')
    logger.info(f'📅 Ejecución: {datetime.utcnow().isoformat()}Z')
    logger.info('=' * 50)

    # Inicializar componentes
    config = Config()

    if not config.validate():
        logger.error('❌ Configuración incompleta. Abortando.')
        sys.exit(1)

    config.ensure_log_dir()

    api = OddsAPI(config)
    model = MLBQuantModel()
    notifier = TelegramNotifier(
        config.get_telegram_token(),
        config.get_telegram_chat_id()
    )
    dedup = DedupManager(config.SENT_LOG_FILE)

    # Limpiar registros de días anteriores
    dedup.cleanup_old_days()

    # Obtener cuotas
    eventos = api.obtener_cuotas()

    if not eventos:
        logger.info('📭 No hay eventos activos para procesar.')
        return

    logger.info(f'📊 Procesando {len(eventos)} eventos...')

    ml_picks = 0
    total_picks = 0
    errors = 0

    for evento in eventos:
        try:
            if process_moneyline(evento, model, dedup, notifier):
                ml_picks += 1
        except Exception as e:
            errors += 1
            logger.error(
                f'❌ Error procesando ML en '
                f'{evento.get("away_team", "?")} @ '
                f'{evento.get("home_team", "?")}: {e}'
            )

        try:
            if process_totals(evento, model, dedup, notifier):
                total_picks += 1
        except Exception as e:
            errors += 1
            logger.error(
                f'❌ Error procesando Totals en '
                f'{evento.get("away_team", "?")} @ '
                f'{evento.get("home_team", "?")}: {e}'
            )

    # Resumen final
    total_sent = ml_picks + total_picks
    logger.info('=' * 50)
    logger.info(
        f'📋 RESUMEN: {ml_picks} ML picks + '
        f'{total_picks} Total picks = '
        f'{total_sent} picks enviados'
    )
    if errors > 0:
        logger.warning(f'⚠️ {errors} errores durante el procesamiento')
    logger.info('=' * 50)

    # Enviar resumen si hubo picks
    if total_sent > 0:
        summary = (
            f"*MLB 2026 - Resumen Horario*\n\n"
            f"⏰ {datetime.utcnow().strftime('%H:%M UTC')}\n"
            f"📊 Eventos procesados: {len(eventos)}\n"
            f"✅ Picks enviados: {total_sent}\n"
            f"  - Moneyline: {ml_picks}\n"
            f"  - Totals: {total_picks}"
        )
        notifier.enviar(summary)
    else:
        logger.info('📭 Sin valor detectado en esta ejecución.')


if __name__ == '__main__':
    run()
