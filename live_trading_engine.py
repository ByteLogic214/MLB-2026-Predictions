"""
═══════════════════════════════════════════════════════════════════════════
live_trading_engine.py — Live Trading Engine v3.0
═══════════════════════════════════════════════════════════════════════════
Sistema de trading automatizado en vivo:
- Detección de arbitraje en tiempo real
- Predicciones in-game con ML
- Gestión de bankroll dinámica
- Alertas instantáneas por Telegram
- Registro de trades ejecutados
═══════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from live_data_fetcher import LiveMLBDataFetcher, LiveFeatureGenerator
from model import ModeloMLBEnsemble
from telegram_notifier import TelegramNotifier
from config import Config
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """
    Motor de trading automatizado para MLB en vivo.
    Ejecuta estrategias basadas en ML + arbitraje.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.25
    ):
        self.config = config or Config()
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.kelly_fraction = kelly_fraction

        # Componentes
        self.fetcher = LiveMLBDataFetcher(config)
        self.feature_gen = LiveFeatureGenerator(self.fetcher)
        self.notifier = TelegramNotifier(config)

        # Modelo ML
        self.model = self._load_model()

        # Registro de trades
        self.trades = []
        self.trade_log_path = './logs/live_trades.json'

        # Umbrales de trading
        self.min_edge = 3.0  # % edge mínimo
        self.max_bet_pct = 5.0  # % máximo del bankroll por apuesta
        self.max_simultaneous_bets = 5

        logger.info(f'🎯 Live Trading Engine inicializado (Bankroll: ${bankroll:,.2f})')

    def _load_model(self) -> ModeloMLBEnsemble:
        """Carga modelo ML entrenado."""
        try:
            model = ModeloMLBEnsemble()
            model.cargar_modelo()
            logger.info('✅ Modelo ML cargado')
            return model
        except Exception as e:
            logger.warning(f'⚠️ No se pudo cargar modelo: {e}')
            return ModeloMLBEnsemble()

    # ═══════════════════════════════════════════════════════════════════
    # DETECCIÓN DE OPORTUNIDADES (Arbitraje + ML Edge)
    # ═══════════════════════════════════════════════════════════════════

    def scan_opportunities(self, date_str: str = '2026-07-09') -> List[Dict]:
        """
        Escanea el mercado buscando oportunidades de valor.
        Returns: Lista de oportunidades con edge positivo.
        """
        logger.info(f'🔍 Escaneando oportunidades para {date_str}...')

        opportunities = []

        try:
            # 1. Obtener juegos en vivo
            live_games = self.fetcher.get_live_games(date_str)

            # 2. Obtener cuotas en vivo
            live_odds = self.fetcher.get_live_odds(date_str)

            if live_odds.empty:
                logger.warning('⚠️ No hay cuotas disponibles')
                return []

            # 3. Para cada juego, calcular edge
            for game in live_games:
                game_opportunities = self._analyze_game(game, live_odds)
                opportunities.extend(game_opportunities)

            # 4. Ordenar por edge descendente
            opportunities.sort(key=lambda x: x['edge'], reverse=True)

            logger.info(f'✅ {len(opportunities)} oportunidades encontradas')
            return opportunities

        except Exception as e:
            logger.error(f'❌ Error escaneando oportunidades: {e}')
            return []

    def _analyze_game(self, game: Dict, odds_df: pd.DataFrame) -> List[Dict]:
        """
        Analiza un juego específico buscando edge.
        """
        opportunities = []

        try:
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']

            # Filtrar cuotas del juego
            game_odds = odds_df[
                (odds_df['home_team'] == home_team) &
                (odds_df['away_team'] == away_team)
            ]

            if game_odds.empty:
                return []

            # 1. MONEYLINE
            ml_opportunities = self._check_moneyline_edge(
                game, game_odds[game_odds['market'] == 'h2h']
            )
            opportunities.extend(ml_opportunities)

            # 2. TOTALS
            totals_opportunities = self._check_totals_edge(
                game, game_odds[game_odds['market'] == 'totals']
            )
            opportunities.extend(totals_opportunities)

            # 3. SPREADS (Run Line)
            spread_opportunities = self._check_spread_edge(
                game, game_odds[game_odds['market'] == 'spreads']
            )
            opportunities.extend(spread_opportunities)

        except Exception as e:
            logger.warning(f'⚠️ Error analizando juego {game.get("game_id")}: {e}')

        return opportunities

    def _check_moneyline_edge(self, game: Dict, odds: pd.DataFrame) -> List[Dict]:
        """Calcula edge en moneyline usando predicción ML."""
        if odds.empty:
            return []

        opportunities = []

        try:
            # Generar features in-game
            features = self.feature_gen.generate_in_game_features(game['game_id'])

            # Predicción ML
            home_prob, away_prob = self._predict_win_probability(features)

            # Obtener mejores cuotas
            home_odds = odds[odds['team'] == game['home_team']]['price'].max()
            away_odds = odds[odds['team'] == game['away_team']]['price'].max()

            # Calcular implied probabilities
            home_implied = self._american_to_probability(home_odds)
            away_implied = self._american_to_probability(away_odds)

            # Calcular edge
            home_edge = (home_prob - home_implied) * 100
            away_edge = (away_prob - away_implied) * 100

            # Agregar si edge > mínimo
            if home_edge > self.min_edge:
                opportunities.append({
                    'game_id': game['game_id'],
                    'type': 'moneyline',
                    'team': game['home_team'],
                    'side': 'home',
                    'predicted_prob': home_prob,
                    'implied_prob': home_implied,
                    'edge': home_edge,
                    'odds': home_odds,
                    'current_score': f"{game['away_score']}-{game['home_score']}",
                    'inning': game['inning'],
                })

            if away_edge > self.min_edge:
                opportunities.append({
                    'game_id': game['game_id'],
                    'type': 'moneyline',
                    'team': game['away_team'],
                    'side': 'away',
                    'predicted_prob': away_prob,
                    'implied_prob': away_implied,
                    'edge': away_edge,
                    'odds': away_odds,
                    'current_score': f"{game['away_score']}-{game['home_score']}",
                    'inning': game['inning'],
                })

        except Exception as e:
            logger.warning(f'⚠️ Error en moneyline edge: {e}')

        return opportunities

    def _check_totals_edge(self, game: Dict, odds: pd.DataFrame) -> List[Dict]:
        """Calcula edge en totals (over/under)."""
        # Implementar lógica similar a moneyline
        # Usar modelo de regresión para predecir total de runs
        return []

    def _check_spread_edge(self, game: Dict, odds: pd.DataFrame) -> List[Dict]:
        """Calcula edge en spreads (run line)."""
        # Implementar lógica similar
        return []

    def _predict_win_probability(self, features: Dict) -> Tuple[float, float]:
        """Predice probabilidad de victoria usando modelo ML."""
        try:
            # Convertir features a formato del modelo
            # (esto depende de cómo esté entrenado tu modelo)

            # Por ahora, mock probabilities
            home_prob = 0.55  # 55%
            away_prob = 0.45  # 45%

            return home_prob, away_prob

        except Exception as e:
            logger.error(f'❌ Error en predicción: {e}')
            return 0.5, 0.5

    # ═══════════════════════════════════════════════════════════════════
    # GESTIÓN DE BANKROLL (Kelly Criterion)
    # ═══════════════════════════════════════════════════════════════════

    def calculate_kelly_stake(
        self,
        probability: float,
        odds: float,
        fraction: float = None
    ) -> float:
        """
        Calcula stake óptimo usando Kelly Criterion.
        
        Args:
            probability: Probabilidad estimada de ganar (0-1)
            odds: Cuota americana
            fraction: Fracción del Kelly a usar (default: self.kelly_fraction)
        
        Returns:
            Monto a apostar en USD
        """
        fraction = fraction or self.kelly_fraction

        try:
            # Convertir odds americanas a decimales
            decimal_odds = self._american_to_decimal(odds)

            # Kelly formula: f = (bp - q) / b
            # donde b = decimal_odds - 1, p = probability, q = 1 - probability
            b = decimal_odds - 1
            p = probability
            q = 1 - p

            kelly_pct = (b * p - q) / b

            # Aplicar fracción (para seguridad)
            kelly_pct *= fraction

            # No apostar si kelly es negativo o muy pequeño
            if kelly_pct <= 0.001:
                return 0.0

            # Calcular stake en USD
            stake = self.bankroll * kelly_pct

            # Aplicar límite máximo
            max_stake = self.bankroll * (self.max_bet_pct / 100)
            stake = min(stake, max_stake)

            return round(stake, 2)

        except Exception as e:
            logger.error(f'❌ Error calculando Kelly: {e}')
            return 0.0

    # ═══════════════════════════════════════════════════════════════════
    # EJECUCIÓN DE TRADES
    # ═══════════════════════════════════════════════════════════════════

    def execute_opportunity(self, opportunity: Dict) -> bool:
        """
        Ejecuta una oportunidad de trading.
        (En producción, esto llamaría a una API de bookmaker)
        """
        try:
            # Calcular stake
            stake = self.calculate_kelly_stake(
                opportunity['predicted_prob'],
                opportunity['odds']
            )

            if stake == 0:
                logger.info(f"⏭️ Stake = $0, skipping {opportunity['team']}")
                return False

            # Verificar límites
            active_bets = len([t for t in self.trades if t['status'] == 'pending'])
            if active_bets >= self.max_simultaneous_bets:
                logger.warning('⚠️ Límite de apuestas simultáneas alcanzado')
                return False

            # Crear registro de trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'game_id': opportunity['game_id'],
                'type': opportunity['type'],
                'team': opportunity['team'],
                'side': opportunity['side'],
                'odds': opportunity['odds'],
                'stake': stake,
                'predicted_prob': opportunity['predicted_prob'],
                'edge': opportunity['edge'],
                'current_score': opportunity['current_score'],
                'inning': opportunity['inning'],
                'status': 'pending',
            }

            # Actualizar bankroll
            self.bankroll -= stake

            # Guardar trade
            self.trades.append(trade)
            self._save_trade_log()

            # Enviar alerta
            self._send_trade_alert(trade)

            logger.info(
                f"✅ TRADE EJECUTADO: {trade['team']} {trade['type']} @ {trade['odds']} "
                f"| Stake: ${stake:,.2f} | Edge: {opportunity['edge']:.2f}%"
            )

            return True

        except Exception as e:
            logger.error(f'❌ Error ejecutando trade: {e}')
            return False

    def _send_trade_alert(self, trade: Dict):
        """Envía alerta de trade por Telegram."""
        try:
            mensaje = (
                f"🎯 **TRADE EN VIVO**\n\n"
                f"**Juego:** {trade['team']}\n"
                f"**Tipo:** {trade['type'].upper()}\n"
                f"**Lado:** {trade['side'].upper()}\n"
                f"**Cuota:** {trade['odds']}\n"
                f"**Stake:** ${trade['stake']:,.2f}\n"
                f"**Edge:** {trade['edge']:.2f}%\n"
                f"**Score:** {trade['current_score']}\n"
                f"**Inning:** {trade['inning']}\n"
                f"**Bankroll restante:** ${self.bankroll:,.2f}"
            )

            self.notifier.enviar_alerta(mensaje)

        except Exception as e:
            logger.warning(f'⚠️ Error enviando alerta: {e}')

    # ═══════════════════════════════════════════════════════════════════
    # MONITOREO Y CIERRE DE TRADES
    # ═══════════════════════════════════════════════════════════════════

    def monitor_active_trades(self):
        """Monitorea trades activos y actualiza su estado."""
        pending_trades = [t for t in self.trades if t['status'] == 'pending']

        for trade in pending_trades:
            result = self._check_trade_result(trade)

            if result == 'won':
                self._close_trade_won(trade)
            elif result == 'lost':
                self._close_trade_lost(trade)

    def _check_trade_result(self, trade: Dict) -> Optional[str]:
        """Verifica si un trade ha sido ganado o perdido."""
        # Consultar resultado del juego
        # En producción, esto consultaría el API en tiempo real

        # Por ahora, mock
        return None  # 'won', 'lost', or None (aún en progreso)

    def _close_trade_won(self, trade: Dict):
        """Cierra un trade ganado."""
        decimal_odds = self._american_to_decimal(trade['odds'])
        profit = trade['stake'] * (decimal_odds - 1)

        self.bankroll += trade['stake'] + profit
        trade['status'] = 'won'
        trade['profit'] = profit
        trade['closed_at'] = datetime.now().isoformat()

        logger.info(f"🎉 TRADE GANADO: ${profit:,.2f} | Bankroll: ${self.bankroll:,.2f}")
        self._save_trade_log()

    def _close_trade_lost(self, trade: Dict):
        """Cierra un trade perdido."""
        trade['status'] = 'lost'
        trade['profit'] = -trade['stake']
        trade['closed_at'] = datetime.now().isoformat()

        logger.info(f"❌ TRADE PERDIDO: -${trade['stake']:,.2f} | Bankroll: ${self.bankroll:,.2f}")
        self._save_trade_log()

    # ═══════════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════

    def run_live_trading(self, date_str: str = '2026-07-09', interval: int = 60):
        """
        Loop principal de trading en vivo.
        
        Args:
            date_str: Fecha objetivo
            interval: Intervalo de escaneo en segundos
        """
        logger.info(f'🚀 Iniciando trading en vivo para {date_str}...')

        try:
            while True:
                # 1. Escanear oportunidades
                opportunities = self.scan_opportunities(date_str)

                # 2. Ejecutar las mejores oportunidades
                for opp in opportunities[:3]:  # Top 3
                    if opp['edge'] > self.min_edge:
                        self.execute_opportunity(opp)

                # 3. Monitorear trades activos
                self.monitor_active_trades()

                # 4. Reportar estado
                self._print_status()

                # 5. Esperar
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info('⏹️ Trading detenido por usuario')
            self._print_final_report()

        except Exception as e:
            logger.error(f'❌ Error en loop de trading: {e}')

    def _print_status(self):
        """Imprime estado actual del trading."""
        pending = len([t for t in self.trades if t['status'] == 'pending'])
        won = len([t for t in self.trades if t['status'] == 'won'])
        lost = len([t for t in self.trades if t['status'] == 'lost'])

        total_profit = sum(t.get('profit', 0) for t in self.trades if t['status'] in ['won', 'lost'])
        roi = (total_profit / self.initial_bankroll) * 100

        logger.info(
            f"📊 STATUS | Bankroll: ${self.bankroll:,.2f} | "
            f"Pending: {pending} | Won: {won} | Lost: {lost} | "
            f"P&L: ${total_profit:,.2f} ({roi:+.2f}%)"
        )

    def _print_final_report(self):
        """Imprime reporte final de trading."""
        print('\n' + '=' * 80)
        print('📈 REPORTE FINAL DE TRADING')
        print('=' * 80)

        total_trades = len(self.trades)
        won = len([t for t in self.trades if t['status'] == 'won'])
        lost = len([t for t in self.trades if t['status'] == 'lost'])
        win_rate = (won / total_trades * 100) if total_trades > 0 else 0

        total_profit = sum(t.get('profit', 0) for t in self.trades if t['status'] in ['won', 'lost'])
        roi = (total_profit / self.initial_bankroll) * 100

        print(f"Bankroll Inicial: ${self.initial_bankroll:,.2f}")
        print(f"Bankroll Final:   ${self.bankroll:,.2f}")
        print(f"P&L Total:        ${total_profit:,.2f} ({roi:+.2f}%)")
        print(f"\nTrades Totales:   {total_trades}")
        print(f"Ganados:          {won}")
        print(f"Perdidos:         {lost}")
        print(f"Win Rate:         {win_rate:.1f}%")
        print('=' * 80 + '\n')

    # ═══════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ═══════════════════════════════════════════════════════════════════

    def _american_to_decimal(self, american_odds: float) -> float:
        """Convierte cuotas americanas a decimales."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def _american_to_probability(self, american_odds: float) -> float:
        """Convierte cuotas americanas a probabilidad implícita."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def _save_trade_log(self):
        """Guarda log de trades en archivo JSON."""
        try:
            import os
            os.makedirs(os.path.dirname(self.trade_log_path), exist_ok=True)

            with open(self.trade_log_path, 'w') as f:
                json.dump(self.trades, f, indent=2)

        except Exception as e:
            logger.warning(f'⚠️ Error guardando trade log: {e}')


# ═══════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO (9 DE JULIO 2026 - EN VIVO)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Inicializar engine con bankroll
    engine = LiveTradingEngine(
        bankroll=10000.0,  # $10k inicial
        kelly_fraction=0.25  # Usar 25% del Kelly (conservador)
    )

    # Ejecutar trading en vivo para el 9 de julio 2026
    engine.run_live_trading(
        date_str='2026-07-09',
        interval=60  # Escanear cada 60 segundos
    )
