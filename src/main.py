"""
═══════════════════════════════════════════════════════════════════════════
main.py — MLB Trading System v4.0 — Live Integration
═══════════════════════════════════════════════════════════════════════════
Sistema completo de trading MLB en vivo:
- Integración con scraper_espn.py + data_pipeline.py
- Live data fetching con actualización continua
- Trading engine automatizado
- Alertas Telegram avanzadas
- Gestión de bankroll con Kelly Criterion
═══════════════════════════════════════════════════════════════════════════
"""

import sys
import logging
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Imports del sistema
from config import Config
from api import OddsAPI
from telegram_notifier import TelegramNotifier
from dedup_manager import DedupManager
from groq_judge import GroqJudge

# Nuevos imports
try:
    from scraper_espn import ESPNScraper
    from data_pipeline import MLBDataPipeline
    from live_data_fetcher import LiveMLBDataFetcher, LiveFeatureGenerator
    from live_trading_engine import LiveTradingEngine
    LIVE_MODULES_AVAILABLE = True
except ImportError as e:
    LIVE_MODULES_AVAILABLE = False
    logging.warning(f'⚠️ Módulos live no disponibles: {e}')

# Model import
try:
    from api import MLBQuantModel  # Modelo mejorado del archivo api.py
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('./logs/mlb_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MLBTradingSystem:
    """
    Sistema completo de trading MLB con capacidades en vivo.
    """

    def __init__(self, mode: str = 'live'):
        """
        Args:
            mode: 'live' (trading en vivo) o 'scan' (escaneo sin ejecutar)
        """
        self.mode = mode
        self.config = Config()
        
        # Validar configuración
        if not self.config.validate():
            logger.error('❌ Configuración inválida')
            sys.exit(1)

        # Inicializar componentes core
        self.odds_api = OddsAPI(self.config)
        self.notifier = TelegramNotifier(
            self.config.get_telegram_token(),
            self.config.get_telegram_chat_id()
        )
        self.dedup = DedupManager(self.config.SENT_LOG_FILE)
        self.judge = GroqJudge(self.config.get_groq_api_key())

        # Inicializar componentes live (si disponibles)
        if LIVE_MODULES_AVAILABLE:
            self.scraper = ESPNScraper()
            self.pipeline = MLBDataPipeline(config=self.config)
            self.live_fetcher = LiveMLBDataFetcher(config=self.config)
            self.feature_gen = LiveFeatureGenerator(self.live_fetcher)
            
            if mode == 'live':
                self.trading_engine = LiveTradingEngine(
                    config=self.config,
                    bankroll=10000.0,  # Configurable
                    kelly_fraction=0.25
                )
        else:
            logger.warning('⚠️ Módulos live no disponibles, usando modo legacy')

        # Modelo ML
        if MODEL_AVAILABLE:
            self.model = MLBQuantModel()
        else:
            logger.warning('⚠️ Modelo ML no disponible')
            self.model = None

        # Estado
        self.is_running = False
        self.opportunities_found = 0
        self.alerts_sent = 0

        logger.info(f'🎯 MLB Trading System v4.0 inicializado (modo: {mode})')

    # ═══════════════════════════════════════════════════════════════════
    # FLUJO PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════

    def run(self, target_date: str = None):
        """
        Ejecuta el flujo principal del sistema.
        
        Args:
            target_date: Fecha objetivo (formato: YYYY-MM-DD)
                        Si es None, usa fecha actual
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f'🚀 Iniciando sistema para {target_date}...')

        try:
            # 1. Enviar notificación de inicio
            self._send_startup_notification(target_date)

            # 2. Actualizar datos (pipeline ETL)
            if LIVE_MODULES_AVAILABLE:
                self._run_data_pipeline()

            # 3. Limpiar logs antiguos
            self.dedup.cleanup_old_days()

            # 4. Ejecutar según modo
            if self.mode == 'live':
                self._run_live_mode(target_date)
            else:
                self._run_scan_mode(target_date)

            # 5. Enviar reporte final
            self._send_final_report()

        except KeyboardInterrupt:
            logger.info('⏹️ Sistema detenido por usuario')
            self._send_shutdown_notification()

        except Exception as e:
            logger.error(f'❌ Error crítico: {e}', exc_info=True)
            self._send_error_alert(str(e))

    def _run_data_pipeline(self):
        """Ejecuta pipeline ETL para actualizar datos."""
        logger.info('📊 Ejecutando pipeline de datos...')
        
        try:
            # Extraer datos de múltiples fuentes
            raw_data = self.pipeline.extract_all_sources(seasons=[2024, 2025])
            
            if not raw_data:
                logger.warning('⚠️ No se extrajeron datos')
                return

            # Transformar y generar features
            features = self.pipeline.transform_to_features(raw_data)
            
            if features.empty:
                logger.warning('⚠️ No se generaron features')
                return

            # Validar calidad
            validation = self.pipeline.validate_features(features)
            
            # Persistir
            self.pipeline.load_features(features, version='production')
            
            logger.info(f'✅ Pipeline completado: {features.shape[0]} registros, '
                       f'{features.shape[1]} features')

            # Reentrenar modelo si es necesario
            if self.model and hasattr(self.model, 'retrain'):
                logger.info('🔄 Reentrenando modelo...')
                self.model.retrain(features)

        except Exception as e:
            logger.error(f'❌ Error en pipeline: {e}')

    # ═══════════════════════════════════════════════════════════════════
    # MODO LIVE (Trading Automatizado)
    # ═══════════════════════════════════════════════════════════════════

    def _run_live_mode(self, target_date: str):
        """Ejecuta trading en vivo con actualización continua."""
        logger.info('🔴 MODO LIVE ACTIVADO')
        
        if not LIVE_MODULES_AVAILABLE:
            logger.error('❌ Módulos live no disponibles')
            return

        # Iniciar updates continuos
        self.live_fetcher.start_live_updates(target_date, interval=30)
        self.is_running = True

        try:
            scan_interval = 60  # Escanear cada 60 segundos
            
            while self.is_running:
                # Escanear oportunidades
                opportunities = self.trading_engine.scan_opportunities(target_date)
                
                if opportunities:
                    logger.info(f'🔍 {len(opportunities)} oportunidades encontradas')
                    
                    # Ejecutar top 3 oportunidades con mejor edge
                    for opp in opportunities[:3]:
                        if self._validate_opportunity(opp):
                            executed = self.trading_engine.execute_opportunity(opp)
                            if executed:
                                self.alerts_sent += 1

                # Monitorear trades activos
                self.trading_engine.monitor_active_trades()

                # Status update cada 5 minutos
                if int(time.time()) % 300 == 0:
                    self._send_status_update()

                time.sleep(scan_interval)

        finally:
            self.live_fetcher.stop_live_updates()
            self.is_running = False

    # ═══════════════════════════════════════════════════════════════════
    # MODO SCAN (Solo Alertas, Sin Trading)
    # ═══════════════════════════════════════════════════════════════════

    def _run_scan_mode(self, target_date: str):
        """Escanea oportunidades y envía alertas sin ejecutar trades."""
        logger.info('🔍 MODO SCAN (solo alertas)')

        # Obtener eventos del día
        if LIVE_MODULES_AVAILABLE:
            eventos = self.live_fetcher.get_live_games(target_date)
            odds_df = self.live_fetcher.get_live_odds(target_date)
        else:
            # Fallback a Odds API
            eventos = self.odds_api.obtener_cuotas()
            odds_df = None

        if not eventos:
            logger.warning('⚠️ No hay eventos disponibles')
            return

        logger.info(f'📋 {len(eventos)} eventos encontrados')

        # Procesar cada evento
        for evento in eventos:
            try:
                self._process_event(evento, odds_df)
            except Exception as e:
                logger.warning(f'⚠️ Error procesando evento: {e}')
                continue

        logger.info(f'✅ Escaneo completado: {self.opportunities_found} oportunidades, '
                   f'{self.alerts_sent} alertas enviadas')

    def _process_event(self, evento: Dict, odds_df=None):
        """Procesa un evento individual buscando oportunidades."""
        
        if isinstance(evento, dict) and 'game_id' in evento:
            # Formato de live_fetcher
            home_team = evento['home_team']
            away_team = evento['away_team']
            game_id = evento['game_id']
        else:
            # Formato de Odds API
            home_team = evento.get('home_team', '')
            away_team = evento.get('away_team', '')
            game_id = None

        # Generar features
        if LIVE_MODULES_AVAILABLE and game_id:
            features = self.feature_gen.generate_in_game_features(game_id)
        else:
            features = self._build_basic_features(home_team, away_team)

        # Procesar mercados
        self._process_moneyline(evento, home_team, away_team, features)
        self._process_totals(evento, home_team, away_team, features)
        self._process_spreads(evento, home_team, away_team, features)

    # ═══════════════════════════════════════════════════════════════════
    # PROCESAMIENTO DE MERCADOS
    # ═══════════════════════════════════════════════════════════════════

    def _process_moneyline(self, evento: Dict, home: str, away: str, features: Dict):
        """Procesa mercado de moneyline."""
        bookmakers = evento.get('bookmakers', [])
        
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') != 'h2h':
                    continue

                for outcome in market.get('outcomes', []):
                    team = outcome.get('name', '')
                    price = outcome.get('price', 0)

                    if price == 0:
                        continue

                    # Predicción
                    if self.model:
                        model_prob = self.model.predict_moneyline(
                            home, away, price, features
                        )
                        edge_data = self.model.calculate_edge(model_prob, price)
                    else:
                        # Fallback básico
                        model_prob = 0.5
                        edge_data = {'edge_pct': 0, 'ev': 0}

                    edge = edge_data['edge_pct']

                    # Filtrar por edge mínimo
                    if edge < 3.0:
                        continue

                    self.opportunities_found += 1

                    # Validar con Groq Judge
                    validation_data = {
                        'market': 'moneyline',
                        'home_team': home,
                        'away_team': away,
                        'team': team,
                        'odds_price': price,
                        'model_prediction': model_prob,
                        'edge': edge,
                    }

                    validation = self.judge.validate_prediction(
                        validation_data, features
                    )

                    if not validation.get('approved', False):
                        logger.debug(f'❌ Groq rechazó: {team} @ {price}')
                        continue

                    # Verificar deduplicación
                    pick_hash = self.dedup.generate_hash(
                        home, away, evento.get('commence_time', ''),
                        'moneyline', team, str(price), bookmaker.get('key', '')
                    )

                    if self.dedup.ya_enviado(pick_hash):
                        continue

                    # Enviar alerta
                    self._send_moneyline_alert(
                        home, away, team, price, edge, edge_data,
                        validation, features
                    )

                    self.dedup.marcar_enviado(pick_hash)
                    self.alerts_sent += 1

    def _process_totals(self, evento: Dict, home: str, away: str, features: Dict):
        """Procesa mercado de totals (over/under)."""
        bookmakers = evento.get('bookmakers', [])
        
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') != 'totals':
                    continue

                for outcome in market.get('outcomes', []):
                    side = outcome.get('name', '')  # 'Over' o 'Under'
                    price = outcome.get('price', 0)
                    point = outcome.get('point', 0)

                    if price == 0 or point == 0:
                        continue

                    # Predicción
                    if self.model:
                        predicted_total = self.model.predict_total(
                            home, away, features
                        )
                        
                        # Calcular probabilidad de over/under
                        if side == 'Over':
                            model_prob = max(0.15, min(0.85, 
                                (predicted_total - point) * 0.1 + 0.5
                            ))
                        else:
                            model_prob = max(0.15, min(0.85,
                                (point - predicted_total) * 0.1 + 0.5
                            ))

                        edge_data = self.model.calculate_edge(model_prob, price)
                    else:
                        continue

                    edge = edge_data['edge_pct']

                    if edge < 3.0:
                        continue

                    self.opportunities_found += 1

                    # Validación y alerta (similar a moneyline)
                    validation_data = {
                        'market': 'totals',
                        'home_team': home,
                        'away_team': away,
                        'side': side,
                        'line': point,
                        'odds_price': price,
                        'predicted_total': predicted_total,
                        'edge': edge,
                    }

                    validation = self.judge.validate_prediction(
                        validation_data, features
                    )

                    if validation.get('approved', False):
                        pick_hash = self.dedup.generate_hash(
                            home, away, evento.get('commence_time', ''),
                            'totals', f'{side}_{point}', str(price),
                            bookmaker.get('key', '')
                        )

                        if not self.dedup.ya_enviado(pick_hash):
                            self._send_totals_alert(
                                home, away, side, point, price, edge,
                                predicted_total, validation
                            )
                            self.dedup.marcar_enviado(pick_hash)
                            self.alerts_sent += 1

    def _process_spreads(self, evento: Dict, home: str, away: str, features: Dict):
        """Procesa mercado de spreads (run line)."""
        # Similar a totals, implementar lógica de spreads
        pass

    # ═══════════════════════════════════════════════════════════════════
    # ALERTAS TELEGRAM MEJORADAS
    # ═══════════════════════════════════════════════════════════════════

    def _send_moneyline_alert(
        self, home: str, away: str, team: str, price: float,
        edge: float, edge_data: Dict, validation: Dict, features: Dict
    ):
        """Envía alerta avanzada de moneyline."""
        
        # Obtener contexto adicional
        current_score = features.get('current_score', 'N/A')
        inning = features.get('inning', 'Pre-game')
        
        # Emoji según edge
        if edge > 8:
            emoji = '🔥🔥🔥'
        elif edge > 5:
            emoji = '🔥🔥'
        else:
            emoji = '🔥'

        mensaje = f"""
{emoji} **MLB MONEYLINE ALERT** {emoji}

**Matchup:** {away} @ {home}
**Pick:** {team} @ {self._format_odds(price)}

**📊 Analysis:**
• Edge: {edge:+.2f}%
• Model Prob: {edge_data['model_prob']*100:.1f}%
• Implied Prob: {edge_data['implied_prob']*100:.1f}%
• EV: {edge_data['ev']:+.4f}
• Kelly: {edge_data['kelly_fraction']*100:.2f}%

**⚾ Game Status:**
• Score: {current_score}
• Inning: {inning}

**🧠 Groq Validation:**
{validation.get('reasoning', 'Approved')}

**💰 Suggested Stake:**
${self._calculate_stake(edge_data['kelly_fraction'])}
(Kelly {edge_data['kelly_fraction']*100:.1f}% @ $10k bankroll)

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.notifier.enviar(mensaje.strip())
        logger.info(f'📨 Alerta enviada: {team} @ {price} (Edge: {edge:.2f}%)')

    def _send_totals_alert(
        self, home: str, away: str, side: str, line: float, price: float,
        edge: float, predicted: float, validation: Dict
    ):
        """Envía alerta de totals."""
        
        emoji = '🎯' if edge > 5 else '📈'
        
        mensaje = f"""
{emoji} **MLB TOTALS ALERT** {emoji}

**Matchup:** {away} @ {home}
**Pick:** {side} {line} @ {self._format_odds(price)}

**📊 Prediction:**
• Predicted Total: {predicted:.1f} runs
• Line: {line}
• Edge: {edge:+.2f}%

**🧠 Groq:** {validation.get('reasoning', 'Approved')}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.notifier.enviar(mensaje.strip())

    def _send_startup_notification(self, target_date: str):
        """Notificación de inicio del sistema."""
        mensaje = f"""
🚀 **MLB TRADING SYSTEM v4.0**

**Status:** INICIANDO
**Fecha:** {target_date}
**Modo:** {self.mode.upper()}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Sistema listo para escanear oportunidades...
"""
        self.notifier.enviar(mensaje.strip())

    def _send_status_update(self):
        """Envía actualización de estado periódica."""
        if self.mode == 'live' and hasattr(self, 'trading_engine'):
            engine = self.trading_engine
            
            pending = len([t for t in engine.trades if t['status'] == 'pending'])
            won = len([t for t in engine.trades if t['status'] == 'won'])
            lost = len([t for t in engine.trades if t['status'] == 'lost'])
            
            total_profit = sum(
                t.get('profit', 0) for t in engine.trades
                if t['status'] in ['won', 'lost']
            )
            roi = (total_profit / engine.initial_bankroll) * 100

            mensaje = f"""
📊 **STATUS UPDATE**

**Bankroll:** ${engine.bankroll:,.2f}
**P&L:** ${total_profit:,.2f} ({roi:+.2f}%)

**Trades:**
• Pending: {pending}
• Won: {won}
• Lost: {lost}

**Session:**
• Opportunities: {self.opportunities_found}
• Alerts Sent: {self.alerts_sent}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            self.notifier.enviar(mensaje.strip())

    def _send_final_report(self):
        """Envía reporte final de la sesión."""
        mensaje = f"""
✅ **SESSION COMPLETED**

**Summary:**
• Opportunities Found: {self.opportunities_found}
• Alerts Sent: {self.alerts_sent}
• Mode: {self.mode.upper()}

"""
        
        if self.mode == 'live' and hasattr(self, 'trading_engine'):
            engine = self.trading_engine
            total_trades = len(engine.trades)
            won = len([t for t in engine.trades if t['status'] == 'won'])
            lost = len([t for t in engine.trades if t['status'] == 'lost'])
            
            total_profit = sum(
                t.get('profit', 0) for t in engine.trades
                if t['status'] in ['won', 'lost']
            )
            roi = (total_profit / engine.initial_bankroll) * 100
            
            mensaje += f"""
**Trading Results:**
• Trades: {total_trades}
• Won: {won}
• Lost: {lost}
• Win Rate: {won/total_trades*100:.1f}%
• Final Bankroll: ${engine.bankroll:,.2f}
• P&L: ${total_profit:,.2f} ({roi:+.2f}%)
"""
        
        mensaje += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.notifier.enviar(mensaje.strip())

    def _send_shutdown_notification(self):
        """Notificación de apagado."""
        mensaje = f"""
⏹️ **SYSTEM SHUTDOWN**

Stopped by user

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.notifier.enviar(mensaje.strip())

    def _send_error_alert(self, error: str):
        """Alerta de error crítico."""
        mensaje = f"""
🚨 **CRITICAL ERROR**

{error}

System may require manual intervention.

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.notifier.enviar(mensaje.strip())

    # ═══════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ═══════════════════════════════════════════════════════════════════

    def _validate_opportunity(self, opp: Dict) -> bool:
        """Valida una oportunidad antes de ejecutar."""
        # Validaciones adicionales de seguridad
        if opp['edge'] < 3.0:
            return False
        
        # Verificar que no esté duplicada
        # (implementar según sea necesario)
        
        return True

    def _build_basic_features(self, home: str, away: str) -> Dict:
        """Construye features básicos cuando no hay datos en vivo."""
        return {
            'home_team': home,
            'away_team': away,
            'current_score': 'N/A',
            'inning': 'Pre-game',
        }

    def _format_odds(self, price: float) -> str:
        """Formatea cuotas americanas."""
        if price > 0:
            return f'+{int(price)}'
        return str(int(price))

    def _calculate_stake(self, kelly_fraction: float, bankroll: float = 10000) -> str:
        """Calcula stake sugerido."""
        stake = bankroll * kelly_fraction
        return f'{stake:,.2f}'


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Punto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MLB Trading System v4.0')
    parser.add_argument(
        '--mode',
        choices=['live', 'scan'],
        default='scan',
        help='Modo de operación (live=trading, scan=solo alertas)'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Fecha objetivo (YYYY-MM-DD), default=hoy'
    )
    
    args = parser.parse_args()
    
    # Crear e iniciar sistema
    system = MLBTradingSystem(mode=args.mode)
    system.run(target_date=args.date)


if __name__ == '__main__':
    main()
