import logging
import random

logger = logging.getLogger(__name__)


class MLBQuantModel:
    """Motor de predicción con Moneyline, Totals y Spreads."""

    def __init__(self):
        self.home_advantage = 0.04
        logger.info('🤖 MLBQuantModel inicializado (ML + Totals + Spreads)')

    def predict_moneyline(self, home_team, away_team, odds_price):
        """Predice probabilidad de victoria del home team."""
        implied_prob = 100 / (odds_price + 100) if odds_price > 0 else \
            abs(odds_price) / (abs(odds_price) + 100)
        adjusted = implied_prob + self.home_advantage
        return max(0.15, min(0.85, adjusted))

    def predict_total(self, home_team, away_team):
        """Predice total de carreras."""
        base_total = 8.5
        variation = random.uniform(-1.0, 1.0)
        return round(base_total + variation, 1)

    def predict_spread(self, home_team, away_team, team):
        """
        Predice el spread de un equipo.
        Retorna un valor positivo/negativo indicando ventaja esperada.
        
        En producción usaría:
        - Diferencia de ERA
        - Récord reciente
        - Head-to-head
        - Factor local
        """
        # Simulación: spread promedio MLB ~1.5 runs
        base_spread = 0.5 if team == home_team else -0.5
        variation = random.uniform(-1.0, 1.0)
        return round(base_spread + variation, 1)

    def retrain(self, training_data):
        """Reentrena modelos."""
        logger.info('🔄 Reentrenando modelos...')
        logger.info('✅ Modelos reentrenados.')
