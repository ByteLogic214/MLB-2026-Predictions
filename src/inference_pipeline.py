"""
inference_pipeline.py — Pipeline de Predicción MLB 2026
Flujo completo: MLB API → Rolling Features → Ensemble → Telegram + CSV
"""
import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'models'))

os.makedirs(os.path.join(_REPO_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(_REPO_DIR, 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(_REPO_DIR, 'data', 'raw'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("InferencePipeline")


# ── 1. Obtener juegos del día vía MLB API ─────────────────────────────────────
def get_mlb_games(date_str: str) -> pd.DataFrame:
    import requests
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        games = []
        for entry in r.json().get('dates', []):
            for g in entry.get('games', []):
                games.append({
                    'game_id':    g.get('gamePk'),
                    'game_date':  date_str,
                    'home_team':  g['teams']['home']['team']['name'],
                    'away_team':  g['teams']['away']['team']['name'],
                    'status':     g.get('status', {}).get('abstractGameState', ''),
                    'home_score': g['teams']['home'].get('score'),
                    'away_score': g['teams']['away'].get('score'),
                })
        df = pd.DataFrame(games)
        logger.info(f"✅ {len(df)} juegos encontrados en MLB API para {date_str}")
        return df
    except Exception as e:
        logger.error(f"❌ Error MLB API: {e}")
        return pd.DataFrame()


# ── 2. Rolling features anti-leakage ─────────────────────────────────────────
def compute_rolling_features(df_history: pd.DataFrame) -> pd.DataFrame:
    """Calcula rolling stats con shift(1) para evitar data leakage."""
    df = df_history.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['away_win'] = 1 - df['home_win']

    for prefix, team_col, score_col, allow_col, win_col in [
        ('home', 'home_team', 'home_score', 'away_score', 'home_win'),
        ('away', 'away_team', 'away_score', 'home_score', 'away_win'),
    ]:
        for w in [5, 10, 25]:
            df[f'{prefix}_runs_scored_L{w}']  = df.groupby(team_col)[score_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df[f'{prefix}_runs_allowed_L{w}'] = df.groupby(team_col)[allow_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df[f'{prefix}_win_pct_L{w}']      = df.groupby(team_col)[win_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    df['run_diff_L5']              = df['home_runs_scored_L5']  - df['away_runs_scored_L5']
    df['run_diff_L10']             = df['home_runs_scored_L10'] - df['away_runs_scored_L10']
    df['pitching_adv_L5']          = df['away_runs_allowed_L5'] - df['home_runs_allowed_L5']
    df['win_pct_diff_L5']          = df['home_win_pct_L5']  - df['away_win_pct_L5']
    df['win_pct_diff_L10']         = df['home_win_pct_L10'] - df['away_win_pct_L10']
    df['win_pct_diff_L25']         = df['home_win_pct_L25'] - df['away_win_pct_L25']
    df['momentum_home']            = df['home_win_pct_L5']  - df['home_win_pct_L25']
    df['momentum_away']            = df['away_win_pct_L5']  - df['away_win_pct_L25']
    df['hot_streak_home']          = (df['home_win_pct_L5'] >= 0.8).astype(int)
    df['hot_streak_away']          = (df['away_win_pct_L5'] >= 0.8).astype(int)
    df['home_advantage']           = 0.035
    df['home_scoring_efficiency']  = df['home_runs_scored_L5'] / (df['home_runs_allowed_L5'] + 0.01)
    df['away_scoring_efficiency']  = df['away_runs_scored_L5'] / (df['away_runs_allowed_L5'] + 0.01)
    df['efficiency_diff']          = df['home_scoring_efficiency'] - df['away_scoring_efficiency']
    return df


def get_latest_team_stats(df_with_features: pd.DataFrame) -> dict:
    """Extrae las últimas stats conocidas por equipo."""
    STAT_COLS = [
        'home_runs_scored_L5','home_runs_scored_L10','home_runs_scored_L25',
        'home_runs_allowed_L5','home_runs_allowed_L10','home_runs_allowed_L25',
        'home_win_pct_L5','home_win_pct_L10','home_win_pct_L25',
        'home_scoring_efficiency','home_advantage',
        'run_diff_L5','run_diff_L10','pitching_adv_L5',
        'win_pct_diff_L5','win_pct_diff_L10','win_pct_diff_L25',
        'momentum_home','hot_streak_home','efficiency_diff',
    ]
    team_latest = {}
    for _, row in df_with_features.sort_values('game_date').iterrows():
        for side in ['home', 'away']:
            team = row[f'{side}_team']
            if team not in team_latest:
                team_latest[team] = {}
            for col in df_with_features.columns:
                if col.startswith(side + '_') or col in ['run_diff_L5','run_diff_L10',
                    'pitching_adv_L5','win_pct_diff_L5','win_pct_diff_L10',
                    'win_pct_diff_L25','momentum_home','momentum_away',
                    'hot_streak_home','hot_streak_away','home_advantage',
                    'efficiency_diff','home_scoring_efficiency','away_scoring_efficiency']:
                    team_latest[team][col] = row.get(col, np.nan)
    return team_latest


def enrich_games(df_games: pd.DataFrame, team_latest: dict) -> pd.DataFrame:
    """Añade features a los juegos de hoy usando las últimas stats conocidas."""
    ALL_FEATS = [
        'home_runs_scored_L5','home_runs_scored_L10','home_runs_scored_L25',
        'home_runs_allowed_L5','home_runs_allowed_L10','home_runs_allowed_L25',
        'home_win_pct_L5','home_win_pct_L10','home_win_pct_L25',
        'away_runs_scored_L5','away_runs_scored_L10','away_runs_scored_L25',
        'away_runs_allowed_L5','away_runs_allowed_L10','away_runs_allowed_L25',
        'away_win_pct_L5','away_win_pct_L10','away_win_pct_L25',
        'run_diff_L5','run_diff_L10','pitching_adv_L5',
        'win_pct_diff_L5','win_pct_diff_L10','win_pct_diff_L25',
        'momentum_home','momentum_away','hot_streak_home','hot_streak_away',
        'home_advantage','efficiency_diff','home_scoring_efficiency','away_scoring_efficiency',
    ]
    df = df_games.copy()
    for feat in ALL_FEATS:
        df[feat] = np.nan

    for idx, row in df.iterrows():
        home_stats = team_latest.get(row['home_team'], {})
        away_stats = team_latest.get(row['away_team'], {})

        for w in [5, 10, 25]:
            df.at[idx, f'home_runs_scored_L{w}']  = home_stats.get(f'home_runs_scored_L{w}', np.nan)
            df.at[idx, f'home_runs_allowed_L{w}']  = home_stats.get(f'home_runs_allowed_L{w}', np.nan)
            df.at[idx, f'home_win_pct_L{w}']       = home_stats.get(f'home_win_pct_L{w}', np.nan)
            df.at[idx, f'away_runs_scored_L{w}']  = away_stats.get(f'away_runs_scored_L{w}', np.nan)
            df.at[idx, f'away_runs_allowed_L{w}']  = away_stats.get(f'away_runs_allowed_L{w}', np.nan)
            df.at[idx, f'away_win_pct_L{w}']       = away_stats.get(f'away_win_pct_L{w}', np.nan)

        # Recalcular diferenciales frescos con los datos de ESTE matchup
        hs5  = df.at[idx, 'home_runs_scored_L5']
        as5  = df.at[idx, 'away_runs_scored_L5']
        hs10 = df.at[idx, 'home_runs_scored_L10']
        as10 = df.at[idx, 'away_runs_scored_L10']
        ha5  = df.at[idx, 'home_runs_allowed_L5']
        aa5  = df.at[idx, 'away_runs_allowed_L5']
        hwp5  = df.at[idx, 'home_win_pct_L5']
        awp5  = df.at[idx, 'away_win_pct_L5']
        hwp10 = df.at[idx, 'home_win_pct_L10']
        awp10 = df.at[idx, 'away_win_pct_L10']
        hwp25 = df.at[idx, 'home_win_pct_L25']
        awp25 = df.at[idx, 'away_win_pct_L25']

        df.at[idx, 'run_diff_L5']      = hs5  - as5  if not (np.isnan(hs5)  or np.isnan(as5))  else np.nan
        df.at[idx, 'run_diff_L10']     = hs10 - as10 if not (np.isnan(hs10) or np.isnan(as10)) else np.nan
        df.at[idx, 'pitching_adv_L5']  = aa5  - ha5  if not (np.isnan(aa5)  or np.isnan(ha5))  else np.nan
        df.at[idx, 'win_pct_diff_L5']  = hwp5  - awp5  if not (np.isnan(hwp5)  or np.isnan(awp5))  else np.nan
        df.at[idx, 'win_pct_diff_L10'] = hwp10 - awp10 if not (np.isnan(hwp10) or np.isnan(awp10)) else np.nan
        df.at[idx, 'win_pct_diff_L25'] = hwp25 - awp25 if not (np.isnan(hwp25) or np.isnan(awp25)) else np.nan
        df.at[idx, 'momentum_home']    = hwp5 - hwp25 if not (np.isnan(hwp5) or np.isnan(hwp25)) else np.nan
        df.at[idx, 'momentum_away']    = awp5 - awp25 if not (np.isnan(awp5) or np.isnan(awp25)) else np.nan
        df.at[idx, 'hot_streak_home']  = int(hwp5 >= 0.8) if not np.isnan(hwp5) else 0
        df.at[idx, 'hot_streak_away']  = int(awp5 >= 0.8) if not np.isnan(awp5) else 0
        df.at[idx, 'home_advantage']   = 0.035

        hse = hs5 / (ha5 + 0.01) if not (np.isnan(hs5) or np.isnan(ha5)) else np.nan
        ase = as5 / (aa5 + 0.01) if not (np.isnan(as5) or np.isnan(aa5)) else np.nan
        df.at[idx, 'home_scoring_efficiency'] = hse
        df.at[idx, 'away_scoring_efficiency'] = ase
        df.at[idx, 'efficiency_diff'] = hse - ase if not (np.isnan(hse) or np.isnan(ase)) else np.nan

    return df


# ── 3. Telegram ───────────────────────────────────────────────────────────────
def send_telegram(mensaje: str) -> bool:
    """Envía mensaje a Telegram. Retorna True si OK."""
    import requests as req
    token   = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip()

    if not token or not chat_id:
        logger.warning("⚠️ Sin TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID — notificación omitida")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Intentar con Markdown
    for parse_mode in ['Markdown', None]:
        try:
            payload = {'chat_id': chat_id, 'text': mensaje, 'disable_web_page_preview': True}
            if parse_mode:
                payload['parse_mode'] = parse_mode
            r = req.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                logger.info("✅ Telegram enviado")
                return True
            else:
                logger.warning(f"⚠️ Telegram HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
            break
    return False


def build_telegram_message(df: pd.DataFrame, date_str: str) -> str:
    """Construye el mensaje de Telegram con picks confiados + resumen."""
    lines = [f"⚾ MLB Predicciones — {date_str}", ""]

    if df.empty:
        lines.append("No hay juegos programados para hoy.")
        return "\n".join(lines)

    confident = df[df.get('confident_pick', pd.Series([False]*len(df))) == True] if 'confident_pick' in df.columns else pd.DataFrame()

    if not confident.empty:
        lines.append(f"🎯 *{len(confident)} PICKS CONFIADOS:*")
        for _, row in confident.iterrows():
            conf = f"{row['prediction_confidence']:.0%}" if 'prediction_confidence' in row else ""
            lines.append(f"• {row['away_team']} @ {row['home_team']}")
            lines.append(f"  → 🏆 *{row['predicted_winner']}* {conf}")
        lines.append("")
    else:
        lines.append("ℹ️ Sin picks de alta confianza hoy.")
        lines.append("(Los 3 modelos no coinciden — partidos equilibrados)")
        lines.append("")

    lines.append(f"📋 Todos los juegos ({len(df)}):")
    for _, row in df.iterrows():
        prob   = row.get('win_probability', 0.5)
        winner = row.get('predicted_winner', '?')
        star   = " 🎯" if row.get('confident_pick', False) else ""
        lines.append(f"• {row['away_team']} @ {row['home_team']}")
        lines.append(f"  → {winner} ({prob:.0%}){star}")

    return "\n".join(lines)


# ── 4. Pipeline principal ─────────────────────────────────────────────────────
def run_prediction_flow(date_str: str = None) -> pd.DataFrame:
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    logger.info("=" * 55)
    logger.info(f"  PIPELINE MLB — {date_str}")
    logger.info("=" * 55)

    # 1. Juegos del día
    df_games = get_mlb_games(date_str)
    if df_games.empty:
        logger.warning("⚠️ Sin juegos para hoy.")
        send_telegram(f"⚾ MLB {date_str}\n\nSin juegos programados para hoy.")
        return pd.DataFrame()

    # Filtrar Preview/Scheduled (no finales)
    to_predict = df_games[df_games['status'].isin(['Preview', 'Pre-Game', 'Scheduled', ''])]
    if to_predict.empty:
        to_predict = df_games
    logger.info(f"  {len(to_predict)} juegos a predecir")

    # 2. Cargar historial y calcular features
    history_path = os.path.join(_REPO_DIR, 'datos_entrenamiento_mlb.csv')
    if not os.path.exists(history_path):
        logger.warning("⚠️ Sin historial. Usando heurística.")
        df_enriched = to_predict.copy()
        df_enriched['home_advantage'] = 0.035
    else:
        df_history = pd.read_csv(history_path)
        df_history = df_history[['game_date','home_team','away_team','home_score','away_score']].dropna()
        df_with_feats = compute_rolling_features(df_history)
        team_latest   = get_latest_team_stats(df_with_feats)
        df_enriched   = enrich_games(to_predict, team_latest)

    # 3. Modelo ensemble (v3 si disponible, v2 como fallback)
    try:
        from models.mlb_ensemble_v3 import MLBEnsembleV3
        model = MLBEnsembleV3(save_dir=os.path.join(_REPO_DIR, 'models'))
        if not model.load():
            raise FileNotFoundError("ensemble_v3.joblib no encontrado")
        logger.info("✅ Usando Ensemble v3 (Poisson+XGB+DT+LR)")
    except Exception as _e:
        logger.warning(f"⚠️ v3 no disponible ({_e}), usando v2")
        from ensemble_model import MLBEnsembleModel
        model = MLBEnsembleModel()

    probs, confident_mask = model.predict_with_confidence(df_enriched, threshold=0.57)
    if not hasattr(confident_mask, '__len__'):
        confident_mask = np.array([confident_mask] * len(df_enriched))

    df_enriched['win_probability']       = probs
    df_enriched['confident_pick']        = confident_mask
    df_enriched['predicted_winner']      = np.where(probs > 0.5, df_enriched['home_team'], df_enriched['away_team'])
    df_enriched['prediction_confidence'] = np.where(probs > 0.5, probs, 1 - probs)

    # 4. Guardar CSV
    out_cols = ['game_id','game_date','home_team','away_team','status',
                'predicted_winner','win_probability','prediction_confidence','confident_pick']
    out_cols = [c for c in out_cols if c in df_enriched.columns]
    out_path = os.path.join(_REPO_DIR, 'data', 'processed', 'predictions_final.csv')
    df_enriched[out_cols].to_csv(out_path, index=False)
    logger.info(f"✅ CSV guardado: {out_path}")

    # 5. Telegram
    msg = build_telegram_message(df_enriched[out_cols], date_str)
    send_telegram(msg)

    # Resumen en log
    conf_count = int(confident_mask.sum()) if hasattr(confident_mask, 'sum') else 0
    logger.info(f"  Juegos: {len(df_enriched)} | Picks confiados: {conf_count}")
    logger.info("=" * 55)

    return df_enriched[out_cols]


if __name__ == "__main__":
    import sys as _sys
    date_arg = _sys.argv[1] if len(_sys.argv) > 1 else None
    run_prediction_flow(date_arg)
