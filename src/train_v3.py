"""
train_v3.py — Reentrenamiento del Ensemble v3 con datos reales
Corre en GitHub Actions donde hay acceso a statsapi y pybaseball.
"""
import sys, os, warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

import pandas as pd
import numpy as np
from datetime import datetime

# ── 1. Descargar datos reales de pitchers ─────────────────────────────────────
def fetch_pitcher_era(seasons=[2023,2024,2025]):
    """Descarga ERA/WHIP/K9 desde pybaseball (FanGraphs)."""
    try:
        from pybaseball import pitching_stats
        dfs = []
        for s in seasons:
            logging.info(f"Descargando pitching {s}...")
            df = pitching_stats(s, qual=20)
            df['season'] = s
            dfs.append(df)
        all_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"✅ {len(all_df)} registros de pitchers descargados")
        return all_df
    except Exception as e:
        logging.warning(f"⚠️ pybaseball error: {e} — usando ERA por defecto")
        return None

def fetch_team_era_from_schedule(seasons=[2023,2024,2025]):
    """Calcula ERA promedio por equipo por temporada desde statsapi."""
    try:
        import statsapi
        team_era = {}
        for season in seasons:
            logging.info(f"Calculando ERA promedio por equipo {season}...")
            teams = statsapi.get('teams', {'sportId':1})['teams']
            for t in teams:
                try:
                    stats = statsapi.get('team_stats', {
                        'teamId': t['id'], 'season': season,
                        'group': 'pitching', 'type': 'season'
                    })
                    era = stats['stats'][0]['splits'][0]['stat'].get('era','4.50')
                    team_era[f"{t['name']}_{season}"] = float(era)
                except: pass
        logging.info(f"✅ ERA de {len(team_era)} equipos obtenida")
        return team_era
    except Exception as e:
        logging.warning(f"⚠️ statsapi error: {e}")
        return {}

# ── 2. Cargar juegos base ─────────────────────────────────────────────────────
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def load_base_games():
    """Carga juegos enriquecidos (con rolling_ERA anti-leakage) vía data_ingestion."""
    from data_ingestion import build_training_dataset, ENRICHED_FILE
    import os
    if os.path.exists(ENRICHED_FILE):
        df = pd.read_csv(ENRICHED_FILE)
        df['date'] = pd.to_datetime(df['date'])
        logging.info(f"✅ {len(df)} juegos desde dataset enriquecido")
        return df
    # Fallback: construir ahora
    logging.info("Construyendo dataset enriquecido...")
    df = build_training_dataset()
    return df

# ── 3. Enriquecer con ERA real ────────────────────────────────────────────────
def enrich_with_era(df, team_era_map):
    """Añade ERA de starter y bullpen estimados por equipo/temporada."""
    def get_era(team, season, default=4.50):
        return team_era_map.get(f"{team}_{season}", default)

    df['season'] = pd.to_datetime(df['date']).dt.year.astype(int)

    for side in ['home','away']:
        team_col = f'{side}_team'
        df[f'{side}_starter_era']  = df.apply(lambda r: get_era(r[team_col], r['season'], 4.50), axis=1)
        df[f'{side}_bullpen_era']  = df.apply(lambda r: get_era(r[team_col], r['season'], 4.20) * 0.95, axis=1)
        df[f'{side}_starter_whip'] = df[f'{side}_starter_era'].apply(lambda e: 0.9 + (e - 3.0) * 0.1)
        df[f'{side}_starter_k9']   = df[f'{side}_starter_era'].apply(lambda e: 10.0 - (e - 3.0) * 0.5)

    return df

# ── 4. Entrenar y guardar ─────────────────────────────────────────────────────
def main():
    logging.info("=== ENTRENAMIENTO Ensemble v3 con datos reales ===")

    # Datos base
    df = load_base_games()

    # ERA real: ya integrada en build_training_dataset via rolling_ERA anti-leakage
    team_era = fetch_team_era_from_schedule([2023,2024,2025])
    if not team_era:
        logging.warning("Usando ERA estimada. Para ERA real, verificar acceso a statsapi.")
        # ERA por defecto por equipo (2023-2025 promedios reales conocidos)
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
        }
        for season in [2023,2024,2025]:
            for team, era in KNOWN_ERA.items():
                team_era[f"{team}_{season}"] = era

    df = enrich_with_era(df, team_era)
    logging.info(f"Muestra de ERA: home_starter_era media={df['home_starter_era'].mean():.2f}")

    # Entrenar
    from models.mlb_ensemble_v3 import MLBEnsembleV3
    model = MLBEnsembleV3(save_dir=os.path.join(REPO_DIR, 'models'))
    metrics = model.train(df)

    logging.info(f"\n✅ ENTRENAMIENTO COMPLETADO")
    logging.info(f"   CV Accuracy : {metrics['cv_accuracy']:.3f}")
    logging.info(f"   Train Acc   : {metrics['train_accuracy']:.3f}")
    logging.info(f"   Features    : {metrics['n_features']}")
    logging.info(f"   Samples     : {metrics['n_samples']}")
    return metrics

if __name__ == '__main__':
    main()
