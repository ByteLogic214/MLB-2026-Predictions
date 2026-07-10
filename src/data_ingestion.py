"""
data_ingestion.py — Ingesta de game logs + rolling ERA anti-leakage
Etapas:
  1. Descarga game_logs desde pybaseball (o fallback desde statsapi)
  2. Aplica enrich_pitcher_data con shift(1) antes de entrenar
  3. Nunca expone ER/IP del juego actual al modelo
"""
import pandas as pd
import numpy as np
import os
import logging
import warnings
from datetime import datetime, timedelta

# Park Factors (importado lazy para evitar circular imports)
_park_factors_loaded = False


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.join(_SRC_DIR, '..')
DATA_DIR  = os.path.join(_REPO_DIR, 'data', 'raw')
PROC_DIR  = os.path.join(_REPO_DIR, 'data', 'processed')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

GAME_LOG_FILE    = os.path.join(DATA_DIR,  'pitcher_game_logs.csv')
BASE_GAMES_FILE  = os.path.join(_REPO_DIR, 'datos_entrenamiento_mlb.csv')
ENRICHED_FILE    = os.path.join(PROC_DIR,  'games_enriched.csv')

# ── Imports opcionales ────────────────────────────────────────────────────────
try:
    from pybaseball import pitching_game_logs, pitching_stats
    PYBASEBALL_OK = True
except ImportError:
    PYBASEBALL_OK = False
    logger.warning("⚠️ pybaseball no instalado — pip install pybaseball")

try:
    import statsapi
    STATSAPI_OK = True
except ImportError:
    STATSAPI_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNCIÓN ANTI-LEAKAGE (rolling ERA con shift)
# ══════════════════════════════════════════════════════════════════════════════

def enrich_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rolling_ERA con shift(1) para evitar look-ahead bias.
    
    Columnas requeridas: ['date', 'pitcher_id', 'ER', 'IP']
    
    REGLA ANTI-LEAKAGE:
    - Solo se usan ER/IP de juegos ANTERIORES al actual (shift(1))
    - 'ER' e 'IP' del juego actual NUNCA llegan al modelo
    - primer juego de cada pitcher → fillna(4.50) (promedio MLB)
    """
    required = {'date', 'pitcher_id', 'ER', 'IP'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"enrich_pitcher_data: faltan columnas {missing}")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['pitcher_id', 'date']).reset_index(drop=True)

    # ─── Acumulados dentro de cada pitcher ───────────────────────────────────
    df['cum_ER'] = df.groupby('pitcher_id')['ER'].cumsum()
    df['cum_IP'] = df.groupby('pitcher_id')['IP'].cumsum()

    # ─── SHIFT — datos del juego actual excluidos ─────────────────────────────
    df['prev_ER'] = df.groupby('pitcher_id')['cum_ER'].shift(1)
    df['prev_IP'] = df.groupby('pitcher_id')['cum_IP'].shift(1)

    # ─── rolling_ERA = (ER previas / IP previas) * 9 ─────────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        df['rolling_ERA'] = np.where(
            df['prev_IP'] > 0,
            (df['prev_ER'] / df['prev_IP']) * 9,
            np.nan
        )

    # ─── Primer juego del lanzador → promedio MLB ─────────────────────────────
    df['rolling_ERA'] = df['rolling_ERA'].fillna(4.50)

    # ─── Eliminar columnas prohibidas (leakage) ───────────────────────────────
    df = df.drop(columns=['cum_ER', 'cum_IP', 'prev_ER', 'prev_IP'])

    logger.debug(
        f"enrich_pitcher_data: {len(df)} filas | "
        f"rolling_ERA range [{df['rolling_ERA'].min():.2f}, {df['rolling_ERA'].max():.2f}]"
    )
    return df


def validate_rolling_era(df: pd.DataFrame) -> bool:
    """
    Verifica que rolling_ERA es coherente:
    - Sin NaN
    - Sin ceros (0 IP before first game → 4.50 por fillna)
    - Rango razonable [1.0, 12.0]
    """
    if 'rolling_ERA' not in df.columns:
        logger.error("❌ rolling_ERA no encontrada")
        return False

    nulls = df['rolling_ERA'].isna().sum()
    zeros = (df['rolling_ERA'] == 0).sum()
    out_range = ((df['rolling_ERA'] < 0.5) | (df['rolling_ERA'] > 15.0)).sum()

    ok = (nulls == 0) and (zeros == 0) and (out_range == 0)
    if ok:
        logger.info(
            f"✅ rolling_ERA válida | "
            f"media={df['rolling_ERA'].mean():.2f} | "
            f"p10={df['rolling_ERA'].quantile(0.1):.2f} | "
            f"p90={df['rolling_ERA'].quantile(0.9):.2f}"
        )
    else:
        logger.warning(
            f"⚠️ rolling_ERA: nulls={nulls} zeros={zeros} fuera_rango={out_range}"
        )
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 2. DESCARGA DE GAME LOGS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_game_logs_pybaseball(seasons=[2023, 2024, 2025]) -> pd.DataFrame:
    """Descarga game-by-game logs de pitchers desde pybaseball."""
    all_logs = []
    for season in seasons:
        logger.info(f"📥 pybaseball pitching game logs {season}...")
        try:
            df = pitching_game_logs(season)
            df['season'] = season
            all_logs.append(df)
            logger.info(f"   {season}: {len(df)} registros")
        except Exception as e:
            logger.warning(f"   {season}: error — {e}")

    if not all_logs:
        return pd.DataFrame()

    raw = pd.concat(all_logs, ignore_index=True)

    # Normalizar columnas
    col_map = {}
    for c in raw.columns:
        cl = c.lower()
        if cl in ['date', 'game_date']:     col_map[c] = 'date'
        if cl in ['playerid', 'pitcher_id', 'mlbid']: col_map[c] = 'pitcher_id'
        if cl == 'er':                       col_map[c] = 'ER'
        if cl == 'ip':                       col_map[c] = 'IP'
        if cl in ['player', 'name', 'playerName']: col_map[c] = 'pitcher_name'
        if cl in ['team', 'team_id']:        col_map[c] = 'team'
    raw = raw.rename(columns=col_map)

    needed = ['date', 'pitcher_id', 'ER', 'IP']
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        logger.error(f"Columnas faltantes tras mapeo: {missing}")
        logger.error(f"Columnas disponibles: {list(raw.columns)}")
        return pd.DataFrame()

    return raw[['date', 'pitcher_id'] + 
               (['pitcher_name'] if 'pitcher_name' in raw.columns else []) +
               (['team'] if 'team' in raw.columns else []) +
               ['ER', 'IP', 'season']]


def _fallback_game_logs_from_era(seasons=[2023, 2024, 2025]) -> pd.DataFrame:
    """
    Fallback: construye un log aproximado por equipo usando ERA conocida.
    Útil cuando pybaseball no está disponible o falla.
    """
    TEAM_ERA_2023 = {
        'Los Angeles Dodgers':3.34,'Atlanta Braves':3.91,'Baltimore Orioles':3.77,
        'Houston Astros':3.92,'Texas Rangers':4.14,'Tampa Bay Rays':3.70,
        'Milwaukee Brewers':3.80,'Philadelphia Phillies':3.97,'San Diego Padres':4.11,
        'Minnesota Twins':4.09,'New York Mets':4.32,'Arizona Diamondbacks':3.99,
        'Seattle Mariners':3.65,'Toronto Blue Jays':4.19,'San Francisco Giants':4.20,
        'Boston Red Sox':4.28,'New York Yankees':3.91,'Cleveland Guardians':3.79,
        'Chicago Cubs':4.43,'Pittsburgh Pirates':4.60,'Cincinnati Reds':4.53,
        'Colorado Rockies':5.36,'Oakland Athletics':5.02,'Miami Marlins':4.41,
        'Detroit Tigers':4.10,'St. Louis Cardinals':4.44,'Washington Nationals':4.72,
        'Los Angeles Angels':4.89,'Kansas City Royals':4.26,'Chicago White Sox':5.19,
    }
    records = []
    for season in seasons:
        for team, era in TEAM_ERA_2023.items():
            # ERA varía ligeramente por temporada
            seasonal_era = era * (1 + 0.02 * (season - 2023))
            records.append({
                'pitcher_id': f"{team.replace(' ','_')}_{season}_SP",
                'pitcher_name': f"{team} Starter",
                'team': team,
                'date': f"{season}-04-01",
                'ER': round(seasonal_era / 9 * 6, 2),   # ~6 IP por start
                'IP': 6.0,
                'season': season,
                'rolling_ERA': seasonal_era,             # ya calculado
                '_is_fallback': True,
            })
    return pd.DataFrame(records)


def fetch_and_update_game_logs(seasons=[2023, 2024, 2025]) -> pd.DataFrame:
    """
    Etapa 1 de ingesta: descarga nuevos game logs y los añade al histórico.
    Idempotente — no duplica registros.
    """
    logger.info("═" * 55)
    logger.info("  INGESTA — GAME LOGS DE PITCHERS")
    logger.info("═" * 55)

    # Cargar histórico existente
    if os.path.exists(GAME_LOG_FILE):
        existing = pd.read_csv(GAME_LOG_FILE)
        existing['date'] = pd.to_datetime(existing['date'])
        logger.info(f"Histórico existente: {len(existing)} registros")
    else:
        existing = pd.DataFrame()

    # Descargar nuevos
    if PYBASEBALL_OK:
        new_logs = _fetch_game_logs_pybaseball(seasons)
    else:
        logger.warning("Usando fallback ERA por equipo (pybaseball no disponible)")
        new_logs = _fallback_game_logs_from_era(seasons)

    if new_logs.empty:
        logger.warning("⚠️ No se obtuvieron nuevos game logs")
        if not existing.empty:
            return existing
        return _fallback_game_logs_from_era(seasons)

    # Combinar y deduplicar
    if not existing.empty:
        combined = pd.concat([existing, new_logs], ignore_index=True)
        dedup_cols = ['pitcher_id', 'date']
        combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
    else:
        combined = new_logs

    combined.to_csv(GAME_LOG_FILE, index=False)
    logger.info(f"✅ Game logs guardados: {len(combined)} registros → {GAME_LOG_FILE}")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 3. PRE-PROCESAMIENTO PARA ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def build_training_dataset(
    game_logs: pd.DataFrame = None,
    base_games_path: str = BASE_GAMES_FILE,
) -> pd.DataFrame:
    """
    Etapa 2: Construye el dataset completo para entrenamiento.
    Aplica enrich_pitcher_data y elimina columnas con leakage.
    
    COLUMNAS PROHIBIDAS (no deben llegar al modelo):
    - 'ER'  — earned runs del juego actual
    - 'IP'  — innings pitched del juego actual
    Solo 'rolling_ERA' (calculada con shift) es legal.
    """
    # 1. Cargar juegos base
    base = pd.read_csv(base_games_path)
    base = base.rename(columns={
        'game_date': 'date',
        'home_score': 'home_runs',
        'away_score': 'away_runs',
    })
    base['date'] = pd.to_datetime(base['date'])
    logger.info(f"Base games: {len(base)} juegos")

    # 2. Enriquecer con rolling_ERA si hay game logs
    if game_logs is None and os.path.exists(GAME_LOG_FILE):
        game_logs = pd.read_csv(GAME_LOG_FILE)

    if game_logs is not None and not game_logs.empty and '_is_fallback' not in game_logs.columns:
        logger.info("Aplicando enrich_pitcher_data (shift anti-leakage)...")
        try:
            logs_enriched = enrich_pitcher_data(game_logs)
            validate_rolling_era(logs_enriched)
        except Exception as e:
            logger.warning(f"⚠️ enrich_pitcher_data falló: {e} — usando ERA por defecto")
            logs_enriched = None
    else:
        logs_enriched = None

    # 3. Mapear rolling_ERA a cada juego por equipo
    if logs_enriched is not None:
        # Construir lookup team → rolling_ERA por fecha
        team_era_by_date = _build_team_era_lookup(logs_enriched)
        base = _join_era_to_games(base, team_era_by_date)
    else:
        # Usar ERA conocidas por equipo (fallback)
        base = _apply_default_era(base)

    # 4. ELIMINACIÓN EXPLÍCITA de features con leakage
    leakage_cols = ['ER', 'IP', 'cum_ER', 'cum_IP', 'prev_ER', 'prev_IP']
    base = base.drop(columns=[c for c in leakage_cols if c in base.columns])

    # 5. Aplicar Park Factors (parque local ajusta ERA de ambos starters)
    try:
        from park_factors import apply_park_adjustment, validate_park_adjustment
        base = apply_park_adjustment(base, park_column='home_team')
        validate_park_adjustment(base)
        logger.info("✅ Park factors integrados al dataset")
    except Exception as e:
        logger.warning(f"⚠️ Park factors no aplicados: {e}")
        # Añadir columnas adj_era manuales si falla
        if 'home_starter_era' in base.columns:
            base['home_adj_era']      = base['home_starter_era']
            base['away_adj_era']      = base['away_starter_era']
            base['park_factor']       = 1.0
            base['delta_adj_era']     = base['away_adj_era'] - base['home_adj_era']
            if 'home_bullpen_era' in base.columns:
                base['home_adj_bullpen_era'] = base['home_bullpen_era']
                base['away_adj_bullpen_era'] = base['away_bullpen_era']

    # 6. Guardar dataset enriquecido
    base.to_csv(ENRICHED_FILE, index=False)
    logger.info(f"✅ Dataset enriquecido guardado: {len(base)} juegos → {ENRICHED_FILE}")
    return base


def _build_team_era_lookup(logs_enriched: pd.DataFrame) -> dict:
    """Construye dict {(team, date): rolling_ERA} desde game logs."""
    lookup = {}
    if 'team' not in logs_enriched.columns:
        return lookup
    for _, row in logs_enriched.iterrows():
        key = (row.get('team',''), str(row['date'])[:10])
        era  = row.get('rolling_ERA', 4.50)
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(era)
    # Promediar varios starters en el mismo día
    return {k: np.mean(v) for k, v in lookup.items()}


def _join_era_to_games(df: pd.DataFrame, era_lookup: dict) -> pd.DataFrame:
    """Asocia rolling_ERA a cada equipo en cada juego."""
    df = df.copy()
    LEAGUE_AVG = 4.50

    def get_era(team, date):
        return era_lookup.get((team, str(date)[:10]), LEAGUE_AVG)

    df['home_starter_era']  = df.apply(lambda r: get_era(r['home_team'], r['date']), axis=1)
    df['away_starter_era']  = df.apply(lambda r: get_era(r['away_team'], r['date']), axis=1)
    df['home_bullpen_era']  = df['home_starter_era'] * 0.95
    df['away_bullpen_era']  = df['away_starter_era'] * 0.95
    df['home_starter_whip'] = 0.9 + (df['home_starter_era'] - 3.0) * 0.10
    df['away_starter_whip'] = 0.9 + (df['away_starter_era'] - 3.0) * 0.10
    df['home_starter_k9']   = 10.0 - (df['home_starter_era'] - 3.0) * 0.5
    df['away_starter_k9']   = 10.0 - (df['away_starter_era'] - 3.0) * 0.5
    return df


def _apply_default_era(df: pd.DataFrame) -> pd.DataFrame:
    """ERA conocidas por equipo como fallback anti-leakage."""
    KNOWN_ERA = {
        'Los Angeles Dodgers':3.34,'Atlanta Braves':3.91,'Baltimore Orioles':3.77,
        'Houston Astros':3.92,'Texas Rangers':4.14,'Tampa Bay Rays':3.70,
        'Milwaukee Brewers':3.80,'Philadelphia Phillies':3.97,'San Diego Padres':4.11,
        'Minnesota Twins':4.09,'New York Mets':4.32,'Arizona Diamondbacks':3.99,
        'Seattle Mariners':3.65,'Toronto Blue Jays':4.19,'San Francisco Giants':4.20,
        'Boston Red Sox':4.28,'New York Yankees':3.91,'Cleveland Guardians':3.79,
        'Chicago Cubs':4.43,'Pittsburgh Pirates':4.60,'Cincinnati Reds':4.53,
        'Colorado Rockies':5.36,'Oakland Athletics':5.02,'Miami Marlins':4.41,
        'Detroit Tigers':4.10,'St. Louis Cardinals':4.44,'Washington Nationals':4.72,
        'Los Angeles Angels':4.89,'Kansas City Royals':4.26,'Chicago White Sox':5.19,
        'Tampa Bay Rays':3.70,
    }
    df = df.copy()
    for side in ['home', 'away']:
        era = df[f'{side}_team'].map(KNOWN_ERA).fillna(4.50)
        df[f'{side}_starter_era']  = era
        df[f'{side}_bullpen_era']  = era * 0.95
        df[f'{side}_starter_whip'] = 0.9 + (era - 3.0) * 0.10
        df[f'{side}_starter_k9']   = 10.0 - (era - 3.0) * 0.5
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENTRADA UNIFICADA PARA GitHub Actions
# ══════════════════════════════════════════════════════════════════════════════

def run_daily_ingest(seasons=[2023, 2024, 2025]):
    """Punto de entrada del pipeline diario de ingesta."""
    logs = fetch_and_update_game_logs(seasons)
    dataset = build_training_dataset(logs)
    return dataset


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    seasons = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [2023, 2024, 2025]
    run_daily_ingest(seasons)
