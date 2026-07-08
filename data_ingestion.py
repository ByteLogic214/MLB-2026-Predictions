"""
data_ingestion.py — Sabermetría Avanzada (corregido para 2026)
"""
import pandas as pd
import numpy as np
import os
import logging
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

try:
    from pybaseball import (
        statcast,
        batting_stats,
        pitching_stats,
        schedule_and_record,
    )
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    logger.warning("⚠️ pybaseball no instalado.")

class DataIngestor:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_path, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        self.batting_file = os.path.join(self.data_dir, 'batting_advanced.csv')
        self.pitching_file = os.path.join(self.data_dir, 'pitching_advanced.csv')
        self.team_stats_file = os.path.join(self.data_dir, 'team_stats_rolling.csv')

        self.season = 2026

    def fetch_batting_advanced(self, min_pa=50):
        logger.info("📊 Extrayendo batting stats...")
        if not PYBASEBALL_AVAILABLE:
            return self._fallback_batting()
        try:
            df = batting_stats(self.season, qual=min_pa)
            # ... (mantén cols_map original)
            df_clean = df  # simplificado
            df_clean.to_csv(self.batting_file, index=False)
            return df_clean
        except Exception as e:
            logger.error(f"❌ Batting error: {e}")
            return self._fallback_batting()

    def fetch_pitching_advanced(self, min_ip=20):
        logger.info("⚾ Extrayendo pitching stats...")
        if not PYBASEBALL_AVAILABLE:
            return self._fallback_pitching()
        try:
            df = pitching_stats(self.season, qual=min_ip)
            df.to_csv(self.pitching_file, index=False)
            return df
        except Exception as e:
            logger.error(f"❌ Pitching error: {e}")
            return self._fallback_pitching()

    def fetch_team_rolling_stats(self):
        logger.info("📈 Team rolling stats...")
        if not PYBASEBALL_AVAILABLE:
            return self._fallback_team_stats()
        try:
            all_teams = []
            teams = ['NYY','BOS','TBR','TOR','BAL','CLE','MIN','CHW','DET','KCR','HOU','SEA','TEX','LAA','OAK','ATL','NYM','PHI','MIA','WSN','MIL','CHC','STL','CIN','PIT','LAD','SDP','SFG','ARI','COL']
            for team in teams:
                try:
                    sched = schedule_and_record(self.season, team)
                    if sched is None or sched.empty: continue
                    sched['runs_scored'] = pd.to_numeric(sched.get('R', 0), errors='coerce')
                    sched['runs_allowed'] = pd.to_numeric(sched.get('RA', 0), errors='coerce')
                    sched['win'] = (sched.get('W/L', '') == 'W').astype(int)
                    for window in [5, 10, 25]:
                        sched[f'runs_scored_L{window}'] = sched['runs_scored'].rolling(window, min_periods=1).mean()
                        sched[f'runs_allowed_L{window}'] = sched['runs_allowed'].rolling(window, min_periods=1).mean()
                        sched[f'win_pct_L{window}'] = sched['win'].rolling(window, min_periods=1).mean()
                    sched['team'] = team
                    all_teams.append(sched)
                except Exception as e:
                    logger.warning(f"⚠️ {team}: {e}")
                    continue
            if all_teams:
                df = pd.concat(all_teams, ignore_index=True)
                df.to_csv(self.team_stats_file, index=False)
                logger.info(f"✅ Team stats: {len(df)} registros")
                return df
        except Exception as e:
            logger.error(f"Team stats error: {e}")
        return self._fallback_team_stats()

    def build_matchup_features(self, home_team, away_team, game_date=None):
        features = {}
        # Team rolling (principal fuente ahora)
        try:
            team_stats = pd.read_csv(self.team_stats_file)
            for prefix, team in [('home', home_team), ('away', away_team)]:
                team_data = team_stats[team_stats['team'] == team]
                if not team_data.empty:
                    latest = team_data.iloc[-1]
                    for w in [5,10,25]:
                        features[f'{prefix}_runs_scored_L{w}'] = latest.get(f'runs_scored_L{w}', np.nan)
                        features[f'{prefix}_runs_allowed_L{w}'] = latest.get(f'runs_allowed_L{w}', np.nan)
                        features[f'{prefix}_win_pct_L{w}'] = latest.get(f'win_pct_L{w}', np.nan)
        except:
            pass
        return features

    def _fallback_batting(self):
        return pd.read_csv(self.batting_file) if os.path.exists(self.batting_file) else pd.DataFrame()

    def _fallback_pitching(self):
        return pd.read_csv(self.pitching_file) if os.path.exists(self.pitching_file) else pd.DataFrame()

    def _fallback_team_stats(self):
        return pd.read_csv(self.team_stats_file) if os.path.exists(self.team_stats_file) else pd.DataFrame()

    def run_daily_update(self):
        logger.info('═' * 60)
        logger.info('  MOTOR DE DATOS — SABERMETRÍA AVANZADA')
        logger.info('═' * 60)

        batting = self.fetch_batting_advanced()
        pitching = self.fetch_pitching_advanced()
        team_stats = self.fetch_team_rolling_stats()

        logger.info(f'  Batting: {len(batting)} | Pitching: {len(pitching)} | Team: {len(team_stats)}')
        logger.info('═' * 60)
        return {'batting': batting, 'pitching': pitching, 'team_stats': team_stats}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DataIngestor().run_daily_update()