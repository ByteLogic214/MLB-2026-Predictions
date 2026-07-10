"""
park_factors.py — Factores de Parque MLB (Park Factors)
Fuente: FanGraphs / Baseball Reference 2023-2025 promedio
Factor > 1.0 → parque favorece bateadores (más carreras)
Factor < 1.0 → parque favorece lanzadores (menos carreras)
1.000 = neutral
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PARK FACTORS — 30 estadios MLB (promedio 2023-2025)
# ══════════════════════════════════════════════════════════════════════════════
PARK_FACTORS = {
    # ── Hitters parks (PF > 1.05) ─────────────────────────────────────────
    'Colorado Rockies':          1.140,  # Coors Field      — altitud/aire seco
    'Cincinnati Reds':           1.075,  # Great American   — viento + dimensiones
    'Texas Rangers':             1.065,  # Globe Life Field — calor + vuelo de pelota
    'Boston Red Sox':            1.060,  # Fenway Park      — Monstruo Verde + tamaño
    'Philadelphia Phillies':     1.055,  # Citizens Bank    — viento del suroeste
    'Chicago Cubs':              1.050,  # Wrigley Field    — viento del lago
    'Atlanta Braves':            1.045,  # Truist Park      — dimensiones cómodas
    'Kansas City Royals':        1.040,  # Kauffman Stadium — outfield amplio, viento
    'Minnesota Twins':           1.035,  # Target Field     — viento + aire frío
    'Baltimore Orioles':         1.030,  # Camden Yards     — RF corto histórico
    'New York Yankees':          1.025,  # Yankee Stadium   — short porch RF
    'Detroit Tigers':            1.020,  # Comerica Park    — RF profundo pero...
    'Pittsburgh Pirates':        1.015,  # PNC Park         — RF corto
    'Houston Astros':            1.010,  # Minute Maid Park — techo, cúpula

    # ── Neutral parks (0.95 - 1.05) ───────────────────────────────────────
    'Chicago White Sox':         1.005,  # Guaranteed Rate
    'Miami Marlins':             1.000,  # loanDepot Park   — techo, neutral
    'Arizona Diamondbacks':      0.995,  # Chase Field      — techo/altitud moderada
    'Toronto Blue Jays':         0.990,  # Rogers Centre    — techo
    'Los Angeles Dodgers':       0.985,  # Dodger Stadium   — costal, neutral
    'Milwaukee Brewers':         0.985,  # Am. Family Field — techo
    'Washington Nationals':      0.980,  # Nationals Park
    'St. Louis Cardinals':       0.975,  # Busch Stadium

    # ── Pitchers parks (PF < 0.95) ────────────────────────────────────────
    'New York Mets':             0.970,  # Citi Field       — amplio, viento marino
    'Cleveland Guardians':       0.965,  # Progressive Field
    'Los Angeles Angels':        0.965,  # Angel Stadium    — dimensiones simétricas
    'San Francisco Giants':      0.960,  # Oracle Park      — viento/niebla marina
    'Seattle Mariners':          0.955,  # T-Mobile Park    — profundo, viento
    'San Diego Padres':          0.950,  # Petco Park       — mayor parque del béisbol
    'Tampa Bay Rays':            0.945,  # Tropicana Field  — artificial, cerrado
    'Oakland Athletics':         0.930,  # Oakland Coliseum — mayor parque, viento
}

# Aliases (abreviaciones → nombre completo para compatibilidad)
TEAM_ABBR_MAP = {
    'COL': 'Colorado Rockies',    'CIN': 'Cincinnati Reds',
    'TEX': 'Texas Rangers',       'BOS': 'Boston Red Sox',
    'PHI': 'Philadelphia Phillies','CHC': 'Chicago Cubs',
    'ATL': 'Atlanta Braves',      'KCR': 'Kansas City Royals',
    'MIN': 'Minnesota Twins',     'BAL': 'Baltimore Orioles',
    'NYY': 'New York Yankees',    'DET': 'Detroit Tigers',
    'PIT': 'Pittsburgh Pirates',  'HOU': 'Houston Astros',
    'CHW': 'Chicago White Sox',   'MIA': 'Miami Marlins',
    'ARI': 'Arizona Diamondbacks','TOR': 'Toronto Blue Jays',
    'LAD': 'Los Angeles Dodgers', 'MIL': 'Milwaukee Brewers',
    'WSN': 'Washington Nationals','STL': 'St. Louis Cardinals',
    'NYM': 'New York Mets',       'CLE': 'Cleveland Guardians',
    'LAA': 'Los Angeles Angels',  'SFG': 'San Francisco Giants',
    'SEA': 'Seattle Mariners',    'SDP': 'San Diego Padres',
    'TBR': 'Tampa Bay Rays',      'OAK': 'Oakland Athletics',
    # Alias adicionales
    'KC':  'Kansas City Royals',  'TB':  'Tampa Bay Rays',
    'SD':  'San Diego Padres',    'SF':  'San Francisco Giants',
    'CWS': 'Chicago White Sox',   'WSH': 'Washington Nationals',
}


def get_park_factor(team_name_or_abbr: str) -> float:
    """
    Retorna el Park Factor para un equipo (nombre o abreviación).
    Default 1.000 si no se encuentra.
    """
    pf = PARK_FACTORS.get(team_name_or_abbr)
    if pf is not None:
        return pf
    full_name = TEAM_ABBR_MAP.get(team_name_or_abbr.upper())
    if full_name:
        return PARK_FACTORS.get(full_name, 1.000)
    return 1.000


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: apply_park_adjustment
# ══════════════════════════════════════════════════════════════════════════════

def apply_park_adjustment(df: pd.DataFrame,
                          park_column: str = 'home_team') -> pd.DataFrame:
    """
    Ajusta rolling_ERA del pitcher por el factor del estadio donde juega HOY.

    REGLA (del consejo técnico):
    - Solo ajustamos por el parque LOCAL (donde se juega el partido)
    - NO ajustamos por el parque visitante
    - Capturamos la ventaja/desventaja real del estadio en ese partido

    Fórmula:
        adj_era = rolling_ERA / park_factor

    Ejemplo:
        ERA 3.50, Coors Field (PF=1.14) → adj_era = 3.50/1.14 = 3.07
        ERA 3.50, Petco Park  (PF=0.95) → adj_era = 3.50/0.95 = 3.68

    Columnas que agrega:
    - park_factor   : factor del parque (para auditoría, se elimina antes de entrenar)
    - home_adj_era  : ERA del starter local ajustado al parque
    - away_adj_era  : ERA del starter visitante ajustado al parque

    Columnas que elimina (features con ruido de estadio):
    - home_starter_era  → reemplazada por home_adj_era
    - away_starter_era  → reemplazada por away_adj_era
    """
    df = df.copy()

    # 1. Factor del parque del partido (SIEMPRE el parque local)
    df['park_factor'] = df[park_column].apply(get_park_factor)

    logger.info(
        f"Park factors — media: {df['park_factor'].mean():.3f} | "
        f"min: {df['park_factor'].min():.3f} ({df.loc[df['park_factor'].idxmin(), park_column]}) | "
        f"max: {df['park_factor'].max():.3f} ({df.loc[df['park_factor'].idxmax(), park_column]})"
    )

    # 2. Ajustar ERA del starter local (lanza en su propio parque)
    if 'home_starter_era' in df.columns:
        df['home_adj_era'] = df['home_starter_era'] / df['park_factor']
    elif 'rolling_ERA' in df.columns:
        df['home_adj_era'] = df['rolling_ERA'] / df['park_factor']
    else:
        df['home_adj_era'] = 4.50 / df['park_factor']

    # 3. Ajustar ERA del starter visitante (lanza en el parque del rival)
    if 'away_starter_era' in df.columns:
        df['away_adj_era'] = df['away_starter_era'] / df['park_factor']
    elif 'rolling_ERA' in df.columns:
        df['away_adj_era'] = df['rolling_ERA'] / df['park_factor']
    else:
        df['away_adj_era'] = 4.50 / df['park_factor']

    # 4. Ajustar también los bullpens al parque
    if 'home_bullpen_era' in df.columns:
        df['home_adj_bullpen_era'] = df['home_bullpen_era'] / df['park_factor']
    if 'away_bullpen_era' in df.columns:
        df['away_adj_bullpen_era'] = df['away_bullpen_era'] / df['park_factor']

    # 5. Ajustar WHIP (correlacionado con ERA)
    if 'home_starter_whip' in df.columns:
        df['home_adj_whip'] = df['home_starter_whip'] / df['park_factor'].apply(lambda pf: 1 + (pf - 1) * 0.6)
    if 'away_starter_whip' in df.columns:
        df['away_adj_whip'] = df['away_starter_whip'] / df['park_factor'].apply(lambda pf: 1 + (pf - 1) * 0.6)

    # 6. Delta ajustado (la feature más potente para XGBoost)
    df['delta_adj_era'] = df['away_adj_era'] - df['home_adj_era']

    # 7. Eliminar columnas crudas (evitar que el modelo vea ERA sin ajuste)
    drop_cols = [c for c in [
        'home_starter_era', 'away_starter_era',
        'home_bullpen_era', 'away_bullpen_era',
        'home_starter_whip', 'away_starter_whip',
        'rolling_ERA',
    ] if c in df.columns]
    df = df.drop(columns=drop_cols)

    logger.info(
        f"✅ Park adjustment aplicado | "
        f"home_adj_era rango: [{df['home_adj_era'].min():.2f}, {df['home_adj_era'].max():.2f}] | "
        f"away_adj_era rango: [{df['away_adj_era'].min():.2f}, {df['away_adj_era'].max():.2f}]"
    )
    return df


def validate_park_adjustment(df: pd.DataFrame) -> bool:
    """Verifica que el ajuste de parque es coherente."""
    checks = {
        'home_adj_era existe':  'home_adj_era' in df.columns,
        'away_adj_era existe':  'away_adj_era' in df.columns,
        'park_factor existe':   'park_factor' in df.columns,
        'sin rolling_ERA cruda': 'rolling_ERA' not in df.columns,
        'sin ERA cruda home':   'home_starter_era' not in df.columns,
        'park_factor rango':    df['park_factor'].between(0.85, 1.20).all() if 'park_factor' in df.columns else False,
        'adj_era sin NaN':      df['home_adj_era'].notna().all() if 'home_adj_era' in df.columns else False,
    }
    all_ok = all(checks.values())
    for check, ok in checks.items():
        status = "✅" if ok else "❌"
        logger.info(f"  {status} {check}")
    return all_ok
