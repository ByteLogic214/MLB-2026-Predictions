"""
inference_pipeline.py — Pipeline de Predicción MLB 2026
Flujo: MLB API → Features → Ensemble → CSV

Accuracy esperado en picks filtrados: ~60-63%
(MLB es el deporte más difícil de predecir; techo real con datos de equipo ~65%)
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajustar paths para imports relativos
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'models'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("InferencePipeline")


def get_mlb_games(date_str: str) -> pd.DataFrame:
    """Obtiene juegos de la API oficial MLB."""
    import requests
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        games = []
        for date_entry in data.get('dates', []):
            for game in date_entry.get('games', []):
                status = game.get('status', {}).get('abstractGameState', '')
                games.append({
                    'game_id': game.get('gamePk'),
                    'game_date': date_str,
                    'home_team': game['teams']['home']['team']['name'],
                    'away_team': game['teams']['away']['team']['name'],
                    'status': status,
                    'home_score': game['teams']['home'].get('score'),
                    'away_score': game['teams']['away'].get('score'),
                })
        return pd.DataFrame(games)
    except Exception as e:
        logger.error(f"❌ Error MLB API: {e}")
        return pd.DataFrame()


def enrich_with_rolling_features(df_games: pd.DataFrame, df_history: pd.DataFrame) -> pd.DataFrame:
    """
    Añade rolling features a los juegos a predecir.
    df_history: historial de juegos pasados con home_score, away_score, home_win
    """
    df_history = df_history.copy()
    df_history['game_date'] = pd.to_datetime(df_history['game_date'])
    df_history['home_win'] = (df_history['home_score'] > df_history['away_score']).astype(int)
    df_history['away_win'] = 1 - df_history['home_win']
    df_history = df_history.sort_values('game_date').reset_index(drop=True)

    def rolling_stats(df, team_col, score_col, allow_col, win_col, prefix):
        stats = {}
        for w in [5, 10, 25]:
            stats[f'{prefix}_runs_scored_L{w}'] = df.groupby(team_col)[score_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            stats[f'{prefix}_runs_allowed_L{w}'] = df.groupby(team_col)[allow_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            stats[f'{prefix}_win_pct_L{w}'] = df.groupby(team_col)[win_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        return stats

    # Calcular stats para todos los registros históricos
    h_stats = rolling_stats(df_history, 'home_team', 'home_score', 'away_score', 'home_win', 'home')
    a_stats = rolling_stats(df_history, 'away_team', 'away_score', 'home_score', 'away_win', 'away')
    for k, v in {**h_stats, **a_stats}.items():
        df_history[k] = v

    # Para cada equipo, tomar las últimas stats conocidas
    team_latest = {}
    for _, row in df_history.sort_values('game_date').iterrows():
        for team_type in ['home', 'away']:
            team = row[f'{team_type}_team']
            team_data = {
                col: row[col] for col in df_history.columns
                if col.startswith(team_type + '_runs') or col.startswith(team_type + '_win_pct')
            }
            team_latest[team] = team_data

    # Aplicar a los juegos de hoy
    enriched_games = df_games.copy()
    for col_prefix, team_col in [('home', 'home_team'), ('away', 'away_team')]:
        for w in [5, 10, 25]:
            for metric in ['runs_scored', 'runs_allowed', 'win_pct']:
                col = f'{col_prefix}_{metric}_L{w}'
                enriched_games[col] = enriched_games[team_col].map(
                    lambda t: team_latest.get(t, {}).get(col, np.nan)
                )

    # Derived features
    enriched_games['run_diff_L5'] = enriched_games['home_runs_scored_L5'] - enriched_games['away_runs_scored_L5']
    enriched_games['run_diff_L10'] = enriched_games['home_runs_scored_L10'] - enriched_games['away_runs_scored_L10']
    enriched_games['pitching_adv_L5'] = enriched_games['away_runs_allowed_L5'] - enriched_games['home_runs_allowed_L5']
    enriched_games['win_pct_diff_L5'] = enriched_games['home_win_pct_L5'] - enriched_games['away_win_pct_L5']
    enriched_games['win_pct_diff_L10'] = enriched_games['home_win_pct_L10'] - enriched_games['away_win_pct_L10']
    enriched_games['win_pct_diff_L25'] = enriched_games['home_win_pct_L25'] - enriched_games['away_win_pct_L25']
    enriched_games['momentum_home'] = enriched_games['home_win_pct_L5'] - enriched_games['home_win_pct_L25']
    enriched_games['momentum_away'] = enriched_games['away_win_pct_L5'] - enriched_games['away_win_pct_L25']
    enriched_games['hot_streak_home'] = (enriched_games['home_win_pct_L5'] >= 0.8).astype(int)
    enriched_games['hot_streak_away'] = (enriched_games['away_win_pct_L5'] >= 0.8).astype(int)
    enriched_games['home_advantage'] = 0.035
    enriched_games['home_scoring_efficiency'] = enriched_games['home_runs_scored_L5'] / (enriched_games['home_runs_allowed_L5'] + 0.01)
    enriched_games['away_scoring_efficiency'] = enriched_games['away_runs_scored_L5'] / (enriched_games['away_runs_allowed_L5'] + 0.01)
    enriched_games['efficiency_diff'] = enriched_games['home_scoring_efficiency'] - enriched_games['away_scoring_efficiency']

    return enriched_games


def run_prediction_flow(date_str: str = None) -> pd.DataFrame:
    """
    Flujo principal de predicción.
    1. Descarga juegos del día
    2. Enriquece con features de historial
    3. Predice con ensemble
    4. Guarda resultados
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"{'='*55}")
    logger.info(f"  PIPELINE DE PREDICCIÓN MLB — {date_str}")
    logger.info(f"{'='*55}")

    # 1. Obtener juegos del día
    logger.info(f"📅 Obteniendo juegos para {date_str}...")
    df_games = get_mlb_games(date_str)

    if df_games.empty:
        logger.warning("⚠️ No hay juegos para esta fecha.")
        return pd.DataFrame()

    preview_games = df_games[df_games['status'].isin(['Preview', 'Pre-Game', 'Scheduled'])]
    if preview_games.empty:
        preview_games = df_games  # Si no hay Preview, predecir todos

    logger.info(f"  {len(preview_games)} juegos a predecir")

    # 2. Cargar historial para features
    history_path = os.path.join(_REPO_DIR, 'datos_entrenamiento_mlb.csv')
    if os.path.exists(history_path):
        logger.info("📊 Cargando historial de juegos...")
        df_history = pd.read_csv(history_path)
        df_games_enriched = enrich_with_rolling_features(preview_games, df_history)
    else:
        logger.warning("⚠️ Sin historial. Usando heurística.")
        df_games_enriched = preview_games.copy()
        df_games_enriched['home_advantage'] = 0.035

    # 3. Cargar modelo y predecir
    from ensemble_model import MLBEnsembleModel
    model = MLBEnsembleModel()

    probs, confident_mask = model.predict_with_confidence(df_games_enriched, threshold=0.57)

    df_games_enriched['win_probability'] = probs
    df_games_enriched['confident_pick'] = confident_mask
    df_games_enriched['predicted_winner'] = np.where(
        probs > 0.5,
        df_games_enriched['home_team'],
        df_games_enriched['away_team']
    )
    df_games_enriched['prediction_confidence'] = np.where(
        probs > 0.5,
        probs,
        1 - probs
    )

    # 4. Guardar resultados
    os.makedirs(os.path.join(_REPO_DIR, 'data', 'processed'), exist_ok=True)
    output_path = os.path.join(_REPO_DIR, 'data', 'processed', 'predictions_final.csv')
    output_cols = ['game_id', 'game_date', 'home_team', 'away_team', 'status',
                   'predicted_winner', 'win_probability', 'prediction_confidence', 'confident_pick']
    available_cols = [c for c in output_cols if c in df_games_enriched.columns]
    df_games_enriched[available_cols].to_csv(output_path, index=False)

    # Resumen
    conf_count = confident_mask.sum() if hasattr(confident_mask, 'sum') else sum(confident_mask)
    logger.info(f"\n{'='*55}")
    logger.info(f"  ✅ PREDICCIONES COMPLETADAS")
    logger.info(f"  Juegos analizados:  {len(df_games_enriched)}")
    logger.info(f"  Picks confiados:    {conf_count}")
    logger.info(f"  Output:             {output_path}")
    logger.info(f"{'='*55}")

    if conf_count > 0:
        picks = df_games_enriched[confident_mask][['away_team','home_team','predicted_winner','prediction_confidence']]
        logger.info("\n🎯 PICKS CONFIADOS:")
        for _, row in picks.iterrows():
            logger.info(f"  {row['away_team']} @ {row['home_team']} → {row['predicted_winner']} ({row['prediction_confidence']:.1%})")

    return df_games_enriched[available_cols]


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    run_prediction_flow(date)


def send_predictions_to_telegram(df_results: pd.DataFrame, date_str: str):
    """
    Envía las predicciones del día a Telegram.
    Solo envía picks confiados. Si no hay picks, manda resumen igual.
    """
    import os
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. Saltando notificación.")
        return

    import requests

    if df_results.empty:
        mensaje = f"⚾ *MLB Predicciones {date_str}*\n\nNo hay juegos programados para hoy."
    else:
        confident = df_results[df_results['confident_pick'] == True] if 'confident_pick' in df_results.columns else pd.DataFrame()

        lineas = [f"⚾ *MLB Predicciones — {date_str}*", ""]

        if not confident.empty:
            lineas.append(f"🎯 *{len(confident)} PICKS CONFIADOS:*")
            for _, row in confident.iterrows():
                conf_pct = f"{row['prediction_confidence']:.0%}" if 'prediction_confidence' in row else ""
                lineas.append(f"• {row['away_team']} @ {row['home_team']}")
                lineas.append(f"  → 🏆 *{row['predicted_winner']}* {conf_pct}")
                lineas.append("")
        else:
            lineas.append("ℹ️ Hoy no hay picks con alta confianza.")
            lineas.append("(Los 3 modelos no coinciden suficientemente)")
            lineas.append("")

        lineas.append(f"📋 *Todos los juegos ({len(df_results)}):*")
        for _, row in df_results.iterrows():
            prob = row.get('win_probability', 0.5)
            winner = row.get('predicted_winner', '?')
            lineas.append(f"• {row['away_team']} @ {row['home_team']} → {winner} ({prob:.0%})")

        mensaje = "\n".join(lineas)

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={
            'chat_id': chat_id,
            'text': mensaje,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }, timeout=15)

        if r.status_code == 200:
            logger.info("✅ Predicciones enviadas a Telegram")
        else:
            logger.error(f"❌ Error Telegram {r.status_code}: {r.text}")
            # Reintentar sin Markdown
            r2 = requests.post(url, json={
                'chat_id': chat_id,
                'text': mensaje.replace('*','').replace('_',''),
                'disable_web_page_preview': True
            }, timeout=15)
            if r2.status_code == 200:
                logger.info("✅ Enviado sin formato")
    except Exception as e:
        logger.error(f"❌ Excepción Telegram: {e}")


# ── Actualizar run_prediction_flow para llamar a Telegram ──────────────
_original_run = run_prediction_flow

def run_prediction_flow(date_str: str = None) -> pd.DataFrame:
    df = _original_run(date_str)
    if date_str is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y-%m-%d')
    send_predictions_to_telegram(df, date_str)
    return df
