import requests
import logging
import json

logger = logging.getLogger(__name__)

GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'


class GroqJudge:
    """
    Juez validador que usa GROQ (Llama 3.3) para:
    1. Verificar la lógica de las predicciones del modelo
    2. Validar edge y consistencia
    3. Sugerir ajustes al modelo si detecta inconsistencias
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.enabled = bool(api_key)
        if not self.enabled:
            logger.warning('⚠️ GROQ Judge deshabilitado (sin API key)')

    def validate_prediction(self, prediction_data):
        """
        Valida una predicción antes de enviarla.
        
        prediction_data debe contener:
        - market: 'moneyline', 'totals', 'spreads'
        - home_team, away_team
        - odds_price
        - model_prediction (probabilidad o total)
        - edge
        
        Retorna:
        - dict con: {'approved': bool, 'confidence': float, 'reasoning': str}
        """
        if not self.enabled:
            return {
                'approved': True,
                'confidence': 0.5,
                'reasoning': 'GROQ Judge deshabilitado'
            }

        prompt = self._build_validation_prompt(prediction_data)

        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'llama-3.3-70b-versatile',
                    'messages': [
                        {
                            'role': 'system',
                            'content': (
                                'Eres un experto en análisis deportivo y betting. '
                                'Validas predicciones de MLB basadas en probabilidades, '
                                'odds y edge. Respondes SOLO en formato JSON.'
                            )
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.3,
                    'max_tokens': 500
                },
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parsear respuesta JSON
                try:
                    validation = json.loads(content)
                    logger.info(
                        f'🧠 GROQ Judge: {validation.get("approved", False)} | '
                        f'Confianza: {validation.get("confidence", 0):.0%}'
                    )
                    return validation
                except json.JSONDecodeError:
                    logger.warning('⚠️ Respuesta GROQ no válida, aprobando por defecto')
                    return {'approved': True, 'confidence': 0.5, 'reasoning': 'Error parsing'}

            else:
                logger.error(f'❌ GROQ API error: {response.status_code}')
                return {'approved': True, 'confidence': 0.5, 'reasoning': 'API error'}

        except requests.exceptions.Timeout:
            logger.warning('⏰ GROQ timeout, aprobando por defecto')
            return {'approved': True, 'confidence': 0.5, 'reasoning': 'Timeout'}
        except Exception as e:
            logger.error(f'❌ Error GROQ Judge: {e}')
            return {'approved': True, 'confidence': 0.5, 'reasoning': str(e)}

    def _build_validation_prompt(self, data):
        """Construye el prompt de validación."""
        market = data.get('market', 'unknown')
        home = data.get('home_team', '?')
        away = data.get('away_team', '?')
        odds = data.get('odds_price', 0)
        prediction = data.get('model_prediction', 0)
        edge = data.get('edge', 0)

        if market == 'moneyline':
            return f"""
Valida esta predicción MLB Moneyline:

**Partido:** {away} @ {home}
**Cuota:** {odds}
**Probabilidad modelo:** {prediction*100:.1f}%
**Edge calculado:** {edge:+.1f}%

Analiza si:
1. El edge es realista (>3% es sospechoso)
2. La probabilidad es coherente con las cuotas
3. Hay suficiente valor para apostar

Responde SOLO con JSON:
{{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "breve explicación"
}}
"""
        elif market == 'totals':
            total_line = data.get('total_line', 0)
            side = data.get('side', '?')
            return f"""
Valida esta predicción MLB Totals:

**Partido:** {away} @ {home}
**Línea:** {side} {total_line}
**Predicción modelo:** {prediction:.1f} runs
**Edge calculado:** {edge:+.1f}%

Analiza si:
1. La predicción es realista (MLB avg ~8.5 runs)
2. El edge justifica la apuesta
3. La diferencia con la línea es significativa

Responde SOLO con JSON:
{{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "breve explicación"
}}
"""
        elif market == 'spreads':
            spread_line = data.get('spread_line', 0)
            return f"""
Valida esta predicción MLB Spread:

**Partido:** {away} @ {home}
**Línea:** {spread_line}
**Predicción modelo:** {prediction:.1f}
**Edge calculado:** {edge:+.1f}%

Analiza si:
1. El spread predicho es realista
2. El edge supera el margen de error típico
3. Vale la pena apostar

Responde SOLO con JSON:
{{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "breve explicación"
}}
"""
        else:
            return '{}'
