"""
═══════════════════════════════════════════════════════════════════════════
groq_judge.py — Validador LLM con Contexto Sabermetrico
═══════════════════════════════════════════════════════════════════════════
Envía diferenciales de ventanas móviles (Delta_xWOBA, Delta_Stuff+, etc.)
al LLM de Groq para validar si el ensemble subestima rachas calientes.
═══════════════════════════════════════════════════════════════════════════
"""
import json
import logging
import requests

logger = logging.getLogger(__name__)

GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'


class GroqJudge:
    """
    Juez LLM que valida predicciones del ensemble usando contexto
    sabermetrico avanzado (deltas de xwOBA, Stuff+, momentum, etc.)
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.enabled = bool(api_key)
        self.model = 'llama-3.3-70b-versatile'

        if not self.enabled:
            logger.warning('⚠️ GROQ Judge deshabilitado (sin API key)')

    def validate_prediction(self, prediction_data, sabermetric_context=None):
        """
        Valida una predicción con contexto sabermetrico.

        prediction_data: dict con market, teams, odds, model_prediction, edge
        sabermetric_context: dict con deltas de métricas avanzadas

        Retorna: {'approved': bool, 'confidence': float, 'reasoning': str,
                  'hot_streak_flag': bool, 'adjusted_edge': float}
        """
        if not self.enabled:
            return {
                'approved': True,
                'confidence': 0.5,
                'reasoning': 'GROQ Judge deshabilitado',
                'hot_streak_flag': False,
                'adjusted_edge': prediction_data.get('edge', 0),
            }

        prompt = self._build_sabermetric_prompt(prediction_data, sabermetric_context)

        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': self.model,
                    'messages': [
                        {
                            'role': 'system',
                            'content': self._system_prompt()
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.2,
                    'max_tokens': 800,
                    'response_format': {'type': 'json_object'}
                },
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                try:
                    validation = json.loads(content)
                    logger.info(
                        f'🧠 GROQ Judge: approved={validation.get("approved", False)} | '
                        f'conf={validation.get("confidence", 0):.0%} | '
                        f'hot_streak={validation.get("hot_streak_flag", False)}'
                    )
                    return validation
                except json.JSONDecodeError:
                    logger.warning('⚠️ Respuesta GROQ no parseable')
                    return self._default_response(prediction_data)
            else:
                logger.error(f'❌ GROQ API error: {response.status_code}')
                return self._default_response(prediction_data)

        except requests.exceptions.Timeout:
            logger.warning('⏰ GROQ timeout')
            return self._default_response(prediction_data)
        except Exception as e:
            logger.error(f'❌ Error GROQ Judge: {e}')
            return self._default_response(prediction_data)

    def _system_prompt(self):
        """System prompt especializado en sabermetría."""
        return """Eres un analista cuantitativo senior de MLB especializado en sabermetría avanzada y detección de valor en apuestas deportivas.

Tu rol es validar predicciones de un modelo Ensemble (XGBoost + LightGBM + RandomForest) usando contexto numérico de métricas avanzadas.

MÉTRICAS QUE RECIBIRÁS:
- Delta_xWOBA: Diferencial de Expected Weighted On-Base Average (positivo = home batting superior)
- Delta_HardHit%: Diferencial de porcentaje de contacto duro
- Delta_Barrel%: Diferencial de barriles (contacto óptimo)
- Delta_Stuff+: Diferencial de calidad física de pitcheos del SP (>100 = above average)
- Delta_Location+: Diferencial de control de zona del SP
- Delta_Pitching+: Diferencial combinado SP
- Delta_Whiff%: Diferencial de tasa de abanicos del SP
- Momentum_L5_vs_L25: Tendencia reciente vs promedio de temporada
- Win_Pct_L5: Win rate últimos 5 juegos de cada equipo

TU TAREA:
1. Evaluar si el edge del modelo es REALISTA dado el contexto sabermetrico
2. Detectar si hay una RACHA CALIENTE (hot streak) que el modelo podría estar subestimando
3. Verificar coherencia entre las métricas y la predicción
4. Ajustar el edge si las métricas sugieren que el modelo está subestimando/sobreestimando

REGLAS:
- Edge > 8% en MLB es MUY sospechoso (rechazar si no hay justificación sabermetrica clara)
- Delta_xWOBA > 0.030 indica ventaja ofensiva significativa
- Delta_Stuff+ > 15 indica ventaja clara del pitcher
- Momentum positivo + Delta_xWOBA positivo = señal fuerte
- Si Win_Pct_L5 > 0.80 para un equipo, hay hot streak activa

Responde SIEMPRE en JSON con esta estructura exacta:
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "explicación breve y técnica",
  "hot_streak_flag": true/false,
  "hot_streak_team": "nombre del equipo o null",
  "adjusted_edge": float (edge corregido),
  "risk_level": "low/medium/high",
  "recommended_kelly": 0.0-0.05
}"""

    def _build_sabermetric_prompt(self, data, context):
        """Construye prompt con contexto sabermetrico completo."""
        market = data.get('market', 'unknown')
        home = data.get('home_team', '?')
        away = data.get('away_team', '?')
        odds = data.get('odds_price', 0)
        prediction = data.get('model_prediction', 0)
        edge = data.get('edge', 0)

        # Base del prompt
        prompt = f"""
═══ VALIDACIÓN DE PREDICCIÓN MLB ═══
Mercado: {market.upper()}
Partido: {away} @ {home}
Cuota: {odds}
"""

        if market == 'moneyline':
            prompt += f"""Probabilidad Modelo: {prediction*100:.1f}%
Edge Calculado: {edge:+.1f}%
"""
        elif market == 'totals':
            total_line = data.get('total_line', 0)
            side = data.get('side', '?')
            prompt += f"""Línea: {side} {total_line}
Predicción Modelo: {prediction:.1f} runs
Edge: {edge:+.1f}%
"""
        elif market == 'spreads':
            spread_line = data.get('spread_line', 0)
            prompt += f"""Spread Línea: {spread_line:+.1f}
Predicción Modelo: {prediction:+.1f}
Edge: {edge:+.1f}%
"""

        # Contexto sabermetrico
        if context:
            prompt += "\n═══ CONTEXTO SABERMETRICO (Deltas Home - Away) ═══\n"

            # Batting deltas
            batting_metrics = {
                'Delta_xWOBA': context.get('delta_xwoba'),
                'Delta_HardHit%': context.get('delta_hard_hit_pct'),
                'Delta_Barrel%': context.get('delta_barrel_pct'),
                'Delta_Chase_Rate': context.get('delta_chase_rate'),
                'Delta_wRC+': context.get('delta_wrc_plus'),
            }
            prompt += "\n📊 BATTING DIFFERENTIALS:\n"
            for name, val in batting_metrics.items():
                if val is not None:
                    direction = "→ HOME superior" if val > 0 else "→ AWAY superior"
                    prompt += f"  {name}: {val:+.4f} {direction}\n"

            # Pitching deltas
            pitching_metrics = {
                'Delta_Stuff+': context.get('delta_sp_stuff_plus'),
                'Delta_Location+': context.get('delta_sp_location_plus'),
                'Delta_Pitching+': context.get('delta_sp_pitching_plus'),
                'Delta_Whiff%': context.get('delta_sp_whiff_pct'),
                'Delta_FB_Velocity': context.get('delta_sp_fb_velocity'),
            }
            prompt += "\n⚾ STARTING PITCHER DIFFERENTIALS:\n"
            for name, val in pitching_metrics.items():
                if val is not None:
                    direction = "→ HOME SP superior" if val > 0 else "→ AWAY SP superior"
                    prompt += f"  {name}: {val:+.2f} {direction}\n"

            # Momentum / Rolling
            prompt += "\n📈 MOMENTUM (L5 vs L25):\n"
            home_momentum = context.get('home_win_pct_L5', 0.5)
            away_momentum = context.get('away_win_pct_L5', 0.5)
            home_rs_l5 = context.get('home_runs_scored_L5', 4.5)
            away_rs_l5 = context.get('away_runs_scored_L5', 4.5)
            home_rs_l25 = context.get('home_runs_scored_L25', 4.5)
            away_rs_l25 = context.get('away_runs_scored_L25', 4.5)

            prompt += f"  {home} Win% L5: {home_momentum:.3f} | Runs L5: {home_rs_l5:.1f} | Runs L25: {home_rs_l25:.1f}\n"
            prompt += f"  {away} Win% L5: {away_momentum:.3f} | Runs L5: {away_rs_l5:.1f} | Runs L25: {away_rs_l25:.1f}\n"

            home_hot = home_momentum >= 0.80
            away_hot = away_momentum >= 0.80
            if home_hot:
                prompt += f"  🔥 {home} EN RACHA CALIENTE (Win% L5 ≥ 80%)\n"
            if away_hot:
                prompt += f"  🔥 {away} EN RACHA CALIENTE (Win% L5 ≥ 80%)\n"

        else:
            prompt += "\n⚠️ Sin contexto sabermetrico disponible. Evaluar solo con odds y edge.\n"

        prompt += """
═══ INSTRUCCIONES ═══
1. ¿El edge es realista dado el contexto sabermetrico?
2. ¿Hay hot streak que el modelo podría subestimar?
3. ¿Las métricas de pitching/batting justifican la predicción?
4. Ajusta el edge si es necesario.

Responde SOLO en JSON."""

        return prompt

    def _default_response(self, data):
        """Respuesta por defecto cuando Groq no está disponible."""
        return {
            'approved': True,
            'confidence': 0.5,
            'reasoning': 'Fallback - sin validación LLM',
            'hot_streak_flag': False,
            'hot_streak_team': None,
            'adjusted_edge': data.get('edge', 0),
            'risk_level': 'medium',
            'recommended_kelly': 0.01,
        }
