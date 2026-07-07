"""
═══════════════════════════════════════════════════════════════════════════
main.py — MLB Quant v8.0 — Sabermetric Ensemble Edition
═══════════════════════════════════════════════════════════════════════════
Integra: data_ingestion + model (ensemble) + groq_judge + telegram
Mercados: Moneyline, Totals, Handicaps (Spreads)
═══════════════════════════════════════════════════════════════════════════
"""
import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Imports locales
from config import Config
from api import OddsAPI
from model import MLBQuantModel
from data_ingestion import DataIngestor
from groq_judge import GroqJudge
from telegram_notifier import TelegramNotifier
from dedup_manager import DedupManager


def process_moneyline(evento, model, dedup, notifier, judge, ingestor):
    """Procesa mercado Moneyline con contexto sabermetrico."""
    home = evento.get('home_team', '')
    away = evento.get('away_team', '')
    commence = evento.get('commence_time', '')

    # Obtener features sabermetricas
    features = ingestor.build_matchup_features(home, away)

    bookmakers = evento.get('bookmakers', [])
    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') != 'h2h':
                continue
            outcomes = market.get('outcomes', [])
            for outcome in outcomes:
                team = outcome.get('name', '')
                price = outcome.get('price', 0)
                if price == 0 or team != home:
                    continue

                pick_hash = dedup.generate_hash(
                    home, away, commence, 'moneyline',
                    team, str(price), bookmaker.get('key', 'unknown')
                )
                if dedup.ya_enviado(pick_hash):
                    continue

                # Predicción con ensemble + features
                model_prob = model.predict_moneyline(home, away, price, features)

                # Calcular edge con Kelly
                edge_data = model.calculate_edge(model_prob, price)
                edge = edge_data['edge_pct']

                logger.info(
                    f'📊 ML: {home} | Prob={model_prob:.1%} | '
                    f'Implied={edge_data["implied_prob"]:.1%} | Edge={edge:+.1f}%'
                )

                # Filtro: edge mínimo 3%
                if edge > 3.0:
                    # Validar con GROQ Judge + contexto sabermetrico
                    validation = judge.validate_prediction(
                        {
                            'market': 'moneyline',
                            'home_team': home,
                            'away_team': away,
                            'odds_price': price,
                            'model_prediction': model_prob,
                            'edge': edge,
                        },
                        sabermetric_context=features
                    )

                    if validation.get('approved', False):
                        confidence = validation.get('confidence', 0.5)
                        reasoning = validation.get('reasoning', '')
                        hot_streak = validation.get('hot_streak_flag', False)
                        adjusted_edge = validation.get('adjusted_edge', edge)
                        kelly = edge_data['kelly_fraction']
                        risk = validation.get('risk_level', 'medium')

                        # Formato del mensaje
                        hot_emoji = "🔥" if hot_streak else ""
                        msg = (
                            f"*MLB 2026 — MONEYLINE* ✅{hot_emoji}\n\n"
                            f"*{away}* @ *{home}*\n"
                            f"📚 {bookmaker.get('title', 'N/A')}\n"
                            f"**Pick: {team}** | Cuota: `{price}`\n\n"
                            f"📊 *Modelo Ensemble:*\n"
                            f"  Prob: `{model_prob*100:.1f}%`\n"
                            f"  Edge: `{adjusted_edge:+.1f}%`\n"
                            f"  EV: `{edge_data['ev']:+.3f}`\n"
                            f"  Kelly: `{kelly*100:.2f}%`\n\n"
                        )

                        # Agregar contexto sabermetrico si disponible
                        if features.get('delta_xwoba') is not None:
                            msg += (
                                f"⚾ *Sabermetrics:*\n"
                                f"  Δ xwOBA: `{features.get('delta_xwoba', 0):+.4f}`\n"
                                f"  Δ Stuff+: `{features.get('delta_sp_stuff_plus', 0):+.1f}`\n"
                                f"  Δ Whiff%: `{features.get('delta_sp_whiff_pct', 0):+.3f}`\n\n"
                            )

                        msg += (
                            f"🧠 *GROQ Judge:* Aprobado ({risk})\n"
                            f"Confianza: `{confidence*100:.0f}%`\n"
                            f"_{reasoning}_\n\n"
                            f"⏰ {commence}"
                        )

                        notifier.enviar(msg)
                        dedup.marcar_enviado(pick_hash)
                        return True

    return False


def process_totals(evento, model, dedup, notifier, judge, ingestor):
    """Procesa mercado de Totals (Over/Under) con contexto sabermetrico."""
    home = evento.get('home_team', '')
    away = evento.get('away_team', '')
    commence = evento.get('commence_time', '')

    features = ingestor.build_matchup_features(home, away)

    bookmakers = evento.get('bookmakers', [])
    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') != 'totals':
                continue
            outcomes = market.get('outcomes', [])
            for outcome in outcomes:
                side = outcome.get('name', '')  # Over/Under
                price = outcome.get('price', 0)
                point = outcome.get('point', 0)
                if price == 0:
                    continue

                pick_hash = dedup.generate_hash(
                    home, away, commence, 'totals',
                    side, str(point), bookmaker.get('key', 'unknown')
                )
                if dedup.ya_enviado(pick_hash):
                    continue

                # Predicción del total
                predicted_total = model.predict_total(home, away, features)

                # Calcular edge
                if side == 'Over':
                    edge = (predicted_total - point) * 8  # ~8% por run de diferencia
                else:
                    edge = (point - predicted_total) * 8

                logger.info(
                    f'📊 Total: {side} {point} | Pred={predicted_total:.1f} | Edge={edge:+.1f}%'
                )

                if abs(edge) > 3.0:
                    validation = judge.validate_prediction(
                        {
                            'market': 'totals',
                            'home_team': home,
                            'away_team': away,
                            'odds_price': price,
                            'model_prediction': predicted_total,
                            'edge': edge,
                            'total_line': point,
                            'side': side,
                        },
                        sabermetric_context=features
                    )

                    if validation.get('approved', False):
                        confidence = validation.get('confidence', 0.5)
                        reasoning = validation.get('reasoning', '')
                        adjusted_edge = validation.get('adjusted_edge', edge)

                        msg = (
                            f"*MLB 2026 — TOTALS* ✅\n\n"
                            f"*{away}* @ *{home}*\n"
                            f"📚 {bookmaker.get('title', 'N/A')}\n"
                            f"**{side} {point}** | Cuota: `{price}`\n\n"
                            f"📊 *Modelo Ensemble:*\n"
                            f"  Predicción: `{predicted_total:.1f} runs`\n"
                            f"  Edge: `{adjusted_edge:+.1f}%`\n\n"
                        )

                        if features.get('home_runs_scored_L5') is not None:
                            msg += (
                                f"📈 *Rolling Stats:*\n"
                                f"  {home} RS L5: `{features.get('home_runs_scored_L5', 0):.1f}`\n"
                                f"  {away} RS L5: `{features.get('away_runs_scored_L5', 0):.1f}`\n"
                                f"  {home} RA L5: `{features.get('home_runs_allowed_L5', 0):.1f}`\n"
                                f"  {away} RA L5: `{features.get('away_runs_allowed_L5', 0):.1f}`\n\n"
                            )

                        msg += (
                            f"🧠 *GROQ Judge:* Aprobado\n"
                            f"Confianza: `{confidence*100:.0f}%`\n"
                            f"_{reasoning}_\n\n"
                            f"⏰ {commence}"
                        )

                        notifier.enviar(msg)
                        dedup.marcar_enviado(pick_hash)
                        return True

    return False


def process_spreads(evento, model, dedup, notifier, judge, ingestor):
    """Procesa mercado de Spreads (Run Line / Handicap)."""
    home = evento.get('home_team', '')
    away = evento.get('away_team', '')
    commence = evento.get('commence_time', '')

    features = ingestor.build_matchup_features(home, away)

    bookmakers = evento.get('bookmakers', [])
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
                    team, str(spread_line), bookmaker.get('key', 'unknown')
                )
                if dedup.ya_enviado(pick_hash):
                    continue

                # Predicción del spread
                predicted_spread = model.predict_spread(home, away, team, features)

                # Edge: diferencia entre spread predicho y línea
                if team == home:
                    edge = (predicted_spread - spread_line) * 5
                else:
                    edge = (predicted_spread - spread_line) * 5

                logger.info(
                    f'📊 Spread: {team} {spread_line:+.1f} | '
                    f'Pred={predicted_spread:+.1f} | Edge={edge:+.1f}%'
                )

                if abs(edge) > 3.0:
                    validation = judge.validate_prediction(
                        {
                            'market': 'spreads',
                            'home_team': home,
                            'away_team': away,
                            'odds_price': price,
                            'model_prediction': predicted_spread,
                            'edge': edge,
                            'spread_line': spread_line,
                        },
                        sabermetric_context=features
                    )

                    if validation.get('approved', False):
                        confidence = validation.get('confidence', 0.5)
                        reasoning = validation.get('reasoning', '')
                        adjusted_edge = validation.get('adjusted_edge', edge)

                        msg = (
                            f"*MLB 2026 — SPREAD/HANDICAP* ✅\n\n"
                            f"*{away}* @ *{home}*\n"
                            f"📚 {bookmaker.get('title', 'N/A')}\n"
                            f"**{team} {spread_line:+.1f}** | Cuota: `{price}`\n\n"
                            f"📊 *Modelo Ensemble:*\n"
                            f"  Spread Pred: `{predicted_spread:+.1f}`\n"
                            f"  Edge: `{adjusted_edge:+.1f}%`\n\n"
                            f"🧠 *GROQ Judge:* Aprobado\n"
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
    logger.info('═' * 60)
    logger.info('  🚀 MLB QUANT v8.0 — SABERMETRIC ENSEMBLE EDITION')
    logger.info(f'  📅 {datetime.utcnow().isoformat()}Z')
    logger.info('═' * 60)

    # Inicializar componentes
    config = Config()
    if not config.validate():
        logger.error('❌ Configuración incompleta.')
        sys.exit(1)
    config.ensure_log_dir()

    api = OddsAPI(config)
    model = MLBQuantModel()
    ingestor = DataIngestor()
    notifier = TelegramNotifier(
        config.get_telegram_token(),
        config.get_telegram_chat_id()
    )
    dedup = DedupManager(config.SENT_LOG_FILE)
    judge = GroqJudge(config.get_groq_api_key())

    # ─── Actualización de datos (si hay tiempo) ──────────────────────
    logger.info('📥 Actualizando datos sabermetricos...')
    try:
        ingestor.run_daily_update()
    except Exception as e:
        logger.warning(f'⚠️ Data update parcial: {e}')

    # ─── Limpiar dedup de días anteriores ────────────────────────────
    dedup.cleanup_old_days()

    # ─── Obtener eventos ─────────────────────────────────────────────
    eventos = api.obtener_cuotas()
    if not eventos:
        logger.info('📭 Sin eventos activos.')
        return

    logger.info(f'📊 Procesando {len(eventos)} eventos...')

    # ─── Procesar cada evento ────────────────────────────────────────
    ml_picks = 0
    total_picks = 0
    spread_picks = 0
    errors = 0

    for evento in eventos:
        try:
            if process_moneyline(evento, model, dedup, notifier, judge, ingestor):
                ml_picks += 1
        except Exception as e:
            errors += 1
            logger.error(f'❌ Error ML: {e}')

        try:
            if process_totals(evento, model, dedup, notifier, judge, ingestor):
                total_picks += 1
        except Exception as e:
            errors += 1
            logger.error(f'❌ Error Totals: {e}')

        try:
            if process_spreads(evento, model, dedup, notifier, judge, ingestor):
                spread_picks += 1
        except Exception as e:
            errors += 1
            logger.error(f'❌ Error Spreads: {e}')

    # ─── Resumen ─────────────────────────────────────────────────────
    total_sent = ml_picks + total_picks + spread_picks
    logger.info('═' * 60)
    logger.info(
        f'  📋 RESUMEN: ML={ml_picks} | Totals={total_picks} | '
        f'Spreads={spread_picks} | TOTAL={total_sent}'
    )
    if errors > 0:
        logger.warning(f'  ⚠️ {errors} errores')
    logger.info('═' * 60)

    if total_sent > 0:
        summary = (
            f"*MLB 2026 — Resumen*\n\n"
            f"⏰ {datetime.utcnow().strftime('%H:%M UTC')}\n"
            f"📊 Eventos: {len(eventos)}\n"
            f"✅ Picks enviados: {total_sent}\n"
            f"  • Moneyline: {ml_picks}\n"
            f"  • Totals: {total_picks}\n"
            f"  • Spreads: {spread_picks}\n\n"
            f"🧠 Validados por GROQ + Sabermetrics\n"
            f"📈 Ensemble: XGBoost + LightGBM + RF"
        )
        notifier.enviar(summary)


if __name__ == '__main__':
    run()

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
