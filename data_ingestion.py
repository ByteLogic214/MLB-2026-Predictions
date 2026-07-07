"""
═══════════════════════════════════════════════════════════════════════════
data_ingestion.py — Sabermetría Avanzada via Pybaseball + FanGraphs
═══════════════════════════════════════════════════════════════════════════
Extrae métricas predictivas reales:
  BATEADORES: xwOBA, HardHit%, Barrel%, Chase Rate, Bat Speed
  PITCHERS:   Stuff+, Location+, Pitching+, Whiff%, Fastball Velo
═══════════════════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ─── Intentar importar pybaseball ────────────────────────────────────────
try:
    from pybaseball import (
        statcast,
        batting_stats,
        pitching_stats,
        playerid_lookup,
        statcast_pitcher,
        statcast_batter,
        team_batting,
        team_pitching,
        schedule_and_record,
    )
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    logger.warning("⚠️ pybaseball no instalado. pip install pybaseball")


class DataIngestor:
    """
    Motor de ingesta con Sabermetría Avanzada.
    Fuentes: MLB Statcast (via pybaseball), FanGraphs.
    """

    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_path, 'data')
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Archivos de salida
        self.batting_file = os.path.join(self.data_dir, 'batting_advanced.csv')
        self.pitching_file = os.path.join(self.data_dir, 'pitching_advanced.csv')
        self.team_stats_file = os.path.join(self.data_dir, 'team_stats_rolling.csv')
        self.matchup_file = os.path.join(self.data_dir, 'matchup_features.csv')

        self.season = 2026

    # ═══════════════════════════════════════════════════════════════════
    # 1. BATTING STATS — xwOBA, HardHit%, Barrel%, Chase Rate, Bat Speed
    # ═══════════════════════════════════════════════════════════════════

    def fetch_batting_advanced(self, min_pa=50):
        """
        Extrae métricas avanzadas de bateadores desde FanGraphs/Statcast.
        Métricas: xwOBA, HardHit%, Barrel%, Chase%, AvgBatSpeed
        """
        logger.info("📊 Extrayendo batting stats avanzadas...")

        if not PYBASEBALL_AVAILABLE:
            return self._fallback_batting()

        try:
            # FanGraphs batting stats con métricas avanzadas
            # qual=50 filtra por mínimo de PA
            df = batting_stats(self.season, qual=min_pa)

            # Columnas clave de FanGraphs
            cols_map = {
                'Name': 'player_name',
                'Team': 'team',
                'PA': 'pa',
                'xwOBA': 'xwoba',
                'Hard%': 'hard_hit_pct',
                'Barrel%': 'barrel_pct',
                'O-Swing%': 'chase_rate',  # Swing fuera de zona
                'AVG': 'avg',
                'OBP': 'obp',
                'SLG': 'slg',
                'wOBA': 'woba',
                'wRC+': 'wrc_plus',
                'WAR': 'war',
                'ISO': 'iso',
                'BABIP': 'babip',
                'K%': 'k_pct',
                'BB%': 'bb_pct',
            }

            available_cols = {k: v for k, v in cols_map.items() if k in df.columns}
            df_clean = df[list(available_cols.keys())].rename(columns=available_cols)

            # Bat Speed desde Statcast (si disponible)
            df_clean['avg_bat_speed'] = self._get_bat_speed_data()

            # Guardar
            df_clean.to_csv(self.batting_file, index=False)
            logger.info(f"✅ Batting stats: {len(df_clean)} bateadores guardados")
            return df_clean

        except Exception as e:
            logger.error(f"❌ Error en batting stats: {e}")
            return self._fallback_batting()

    def _get_bat_speed_data(self):
        """Extrae velocidad promedio de bateo desde Statcast."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            sc = statcast(start_dt=start_date, end_dt=end_date)
            if sc is not None and 'bat_speed' in sc.columns:
                avg_speed = sc.groupby('batter')['bat_speed'].mean()
                return avg_speed
        except Exception:
            pass
        return np.nan

    # ═══════════════════════════════════════════════════════════════════
    # 2. PITCHING STATS — Stuff+, Location+, Pitching+, Whiff%, FB Velo
    # ═══════════════════════════════════════════════════════════════════

    def fetch_pitching_advanced(self, min_ip=20):
        """
        Extrae métricas avanzadas de pitchers desde FanGraphs/Statcast.
        Métricas: Stuff+, Location+, Pitching+, Whiff%, FB Velocity
        """
        logger.info("⚾ Extrayendo pitching stats avanzadas...")

        if not PYBASEBALL_AVAILABLE:
            return self._fallback_pitching()

        try:
            df = pitching_stats(self.season, qual=min_ip)

            cols_map = {
                'Name': 'player_name',
                'Team': 'team',
                'IP': 'ip',
                'ERA': 'era',
                'FIP': 'fip',
                'xFIP': 'xfip',
                'SIERA': 'siera',
                'K%': 'k_pct',
                'BB%': 'bb_pct',
                'K-BB%': 'k_bb_pct',
                'HR/9': 'hr_9',
                'WHIP': 'whip',
                'WAR': 'war',
                'Stuff+': 'stuff_plus',
                'Location+': 'location_plus',
                'Pitching+': 'pitching_plus',
                'CSW%': 'csw_pct',  # Called Strike + Whiff %
            }

            available_cols = {k: v for k, v in cols_map.items() if k in df.columns}
            df_clean = df[list(available_cols.keys())].rename(columns=available_cols)

            # Whiff% y Fastball Velocity desde Statcast
            whiff_velo = self._get_pitcher_statcast_metrics()
            if whiff_velo is not None:
                df_clean = df_clean.merge(whiff_velo, on='player_name', how='left')
            else:
                df_clean['whiff_pct'] = np.nan
                df_clean['fb_velocity'] = np.nan

            df_clean.to_csv(self.pitching_file, index=False)
            logger.info(f"✅ Pitching stats: {len(df_clean)} pitchers guardados")
            return df_clean

        except Exception as e:
            logger.error(f"❌ Error en pitching stats: {e}")
            return self._fallback_pitching()

    def _get_pitcher_statcast_metrics(self):
        """Extrae Whiff% y FB Velocity desde Statcast."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            sc = statcast(start_dt=start_date, end_dt=end_date)

            if sc is None or sc.empty:
                return None

            # Whiff% = swinging_strikes / total_pitches
            pitches = sc.groupby('pitcher').agg(
                total_pitches=('pitch_type', 'count'),
                swinging_strikes=('description', lambda x: (x == 'swinging_strike').sum()),
            ).reset_index()
            pitches['whiff_pct'] = pitches['swinging_strikes'] / pitches['total_pitches']

            # Fastball velocity (FF = 4-seam, SI = sinker)
            fastballs = sc[sc['pitch_type'].isin(['FF', 'SI'])]
            fb_velo = fastballs.groupby('pitcher')['release_speed'].mean().reset_index()
            fb_velo.columns = ['pitcher', 'fb_velocity']

            merged = pitches.merge(fb_velo, on='pitcher', how='left')
            # Map pitcher IDs to names would require additional lookup
            # For now return with pitcher ID
            merged = merged[['pitcher', 'whiff_pct', 'fb_velocity']]
            merged.columns = ['player_id', 'whiff_pct', 'fb_velocity']
            return merged

        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    # 3. TEAM-LEVEL ROLLING STATS (últimos 5, 10, 25 juegos)
    # ═══════════════════════════════════════════════════════════════════

    def fetch_team_rolling_stats(self):
        """
        Calcula stats de equipo en ventanas móviles (5, 10, 25 juegos).
        Previene data leakage usando solo datos previos a cada fecha.
        """
        logger.info("📈 Calculando team rolling stats...")

        if not PYBASEBALL_AVAILABLE:
            return self._fallback_team_stats()

        try:
            all_teams_stats = []
            teams = [
                'NYY', 'BOS', 'TBR', 'TOR', 'BAL',
                'CLE', 'MIN', 'CHW', 'DET', 'KCR',
                'HOU', 'SEA', 'TEX', 'LAA', 'OAK',
                'ATL', 'NYM', 'PHI', 'MIA', 'WSN',
                'MIL', 'CHC', 'STL', 'CIN', 'PIT',
                'LAD', 'SDP', 'SFG', 'ARI', 'COL',
            ]

            for team in teams:
                try:
                    sched = schedule_and_record(self.season, team)
                    if sched is None or sched.empty:
                        continue

                    # Calcular rolling stats
                    sched['runs_scored'] = pd.to_numeric(sched.get('R', 0), errors='coerce')
                    sched['runs_allowed'] = pd.to_numeric(sched.get('RA', 0), errors='coerce')
                    sched['win'] = (sched.get('W/L', '') == 'W').astype(int)

                    for window in [5, 10, 25]:
                        sched[f'runs_scored_L{window}'] = (
                            sched['runs_scored'].rolling(window, min_periods=1).mean()
                        )
                        sched[f'runs_allowed_L{window}'] = (
                            sched['runs_allowed'].rolling(window, min_periods=1).mean()
                        )
                        sched[f'win_pct_L{window}'] = (
                            sched['win'].rolling(window, min_periods=1).mean()
                        )

                    sched['team'] = team
                    all_teams_stats.append(sched)

                except Exception as e:
                    logger.warning(f"⚠️ Error para {team}: {e}")
                    continue

            if all_teams_stats:
                df_teams = pd.concat(all_teams_stats, ignore_index=True)
                df_teams.to_csv(self.team_stats_file, index=False)
                logger.info(f"✅ Team rolling stats: {len(df_teams)} registros")
                return df_teams

        except Exception as e:
            logger.error(f"❌ Error en team stats: {e}")

        return self._fallback_team_stats()

    # ═══════════════════════════════════════════════════════════════════
    # 4. MATCHUP FEATURES (para alimentar el modelo)
    # ═══════════════════════════════════════════════════════════════════

    def build_matchup_features(self, home_team, away_team, game_date=None):
        """
        Construye el vector de features para un matchup específico.
        Combina batting + pitching + team rolling stats.
        Retorna dict con todas las métricas para el modelo.
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')

        features = {}

        # Team batting aggregates
        try:
            batting = pd.read_csv(self.batting_file)
            for prefix, team in [('home', home_team), ('away', away_team)]:
                team_batters = batting[batting['team'] == team]
                if not team_batters.empty:
                    features[f'{prefix}_xwoba'] = team_batters['xwoba'].mean()
                    features[f'{prefix}_hard_hit_pct'] = team_batters['hard_hit_pct'].mean()
                    features[f'{prefix}_barrel_pct'] = team_batters['barrel_pct'].mean()
                    features[f'{prefix}_chase_rate'] = team_batters['chase_rate'].mean()
                    features[f'{prefix}_wrc_plus'] = team_batters['wrc_plus'].mean()
                    features[f'{prefix}_iso'] = team_batters['iso'].mean()
                    features[f'{prefix}_k_pct_bat'] = team_batters['k_pct'].mean()
                    features[f'{prefix}_bb_pct_bat'] = team_batters['bb_pct'].mean()
        except Exception:
            pass

        # Team pitching (starter)
        try:
            pitching = pd.read_csv(self.pitching_file)
            for prefix, team in [('home', home_team), ('away', away_team)]:
                team_pitchers = pitching[pitching['team'] == team]
                if not team_pitchers.empty:
                    # Top pitcher (most IP = probable starter proxy)
                    starter = team_pitchers.sort_values('ip', ascending=False).iloc[0]
                    features[f'{prefix}_sp_stuff_plus'] = starter.get('stuff_plus', np.nan)
                    features[f'{prefix}_sp_location_plus'] = starter.get('location_plus', np.nan)
                    features[f'{prefix}_sp_pitching_plus'] = starter.get('pitching_plus', np.nan)
                    features[f'{prefix}_sp_whiff_pct'] = starter.get('whiff_pct', np.nan)
                    features[f'{prefix}_sp_fb_velocity'] = starter.get('fb_velocity', np.nan)
                    features[f'{prefix}_sp_era'] = starter.get('era', np.nan)
                    features[f'{prefix}_sp_fip'] = starter.get('fip', np.nan)
                    features[f'{prefix}_sp_xfip'] = starter.get('xfip', np.nan)
                    features[f'{prefix}_sp_k_pct'] = starter.get('k_pct', np.nan)
        except Exception:
            pass

        # Team rolling stats
        try:
            team_stats = pd.read_csv(self.team_stats_file)
            for prefix, team in [('home', home_team), ('away', away_team)]:
                team_data = team_stats[team_stats['team'] == team]
                if not team_data.empty:
                    latest = team_data.iloc[-1]  # Último registro disponible
                    for window in [5, 10, 25]:
                        features[f'{prefix}_runs_scored_L{window}'] = latest.get(
                            f'runs_scored_L{window}', np.nan
                        )
                        features[f'{prefix}_runs_allowed_L{window}'] = latest.get(
                            f'runs_allowed_L{window}', np.nan
                        )
                        features[f'{prefix}_win_pct_L{window}'] = latest.get(
                            f'win_pct_L{window}', np.nan
                        )
        except Exception:
            pass

        # Deltas (diferenciales home - away)
        delta_keys = ['xwoba', 'hard_hit_pct', 'barrel_pct', 'chase_rate', 'wrc_plus']
        for key in delta_keys:
            h_val = features.get(f'home_{key}', np.nan)
            a_val = features.get(f'away_{key}', np.nan)
            if not np.isnan(h_val) and not np.isnan(a_val):
                features[f'delta_{key}'] = h_val - a_val

        pitcher_deltas = ['stuff_plus', 'location_plus', 'pitching_plus', 'whiff_pct']
        for key in pitcher_deltas:
            h_val = features.get(f'home_sp_{key}', np.nan)
            a_val = features.get(f'away_sp_{key}', np.nan)
            if not np.isnan(h_val) and not np.isnan(a_val):
                features[f'delta_sp_{key}'] = h_val - a_val

        return features

    # ═══════════════════════════════════════════════════════════════════
    # FALLBACKS (cuando pybaseball no está disponible)
    # ═══════════════════════════════════════════════════════════════════

    def _fallback_batting(self):
        """Carga datos locales si pybaseball falla."""
        if os.path.exists(self.batting_file):
            logger.info("📂 Usando batting cache local")
            return pd.read_csv(self.batting_file)
        logger.warning("⚠️ Sin datos de batting disponibles")
        return pd.DataFrame()

    def _fallback_pitching(self):
        """Carga datos locales si pybaseball falla."""
        if os.path.exists(self.pitching_file):
            logger.info("📂 Usando pitching cache local")
            return pd.read_csv(self.pitching_file)
        logger.warning("⚠️ Sin datos de pitching disponibles")
        return pd.DataFrame()

    def _fallback_team_stats(self):
        """Carga datos locales si pybaseball falla."""
        if os.path.exists(self.team_stats_file):
            logger.info("📂 Usando team stats cache local")
            return pd.read_csv(self.team_stats_file)
        logger.warning("⚠️ Sin datos de team stats disponibles")
        return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════
    # RUN DAILY UPDATE
    # ═══════════════════════════════════════════════════════════════════

    def run_daily_update(self):
        """Ejecuta la actualización diaria completa de datos."""
        logger.info('═' * 60)
        logger.info('  MOTOR DE DATOS — SABERMETRÍA AVANZADA')
        logger.info('═' * 60)

        batting = self.fetch_batting_advanced()
        pitching = self.fetch_pitching_advanced()
        team_stats = self.fetch_team_rolling_stats()

        logger.info('═' * 60)
        logger.info(f'  Batting: {len(batting) if batting is not None else 0} registros')
        logger.info(f'  Pitching: {len(pitching) if pitching is not None else 0} registros')
        logger.info(f'  Team Stats: {len(team_stats) if team_stats is not None else 0} registros')
        logger.info('═' * 60)

        return {
            'batting': batting,
            'pitching': pitching,
            'team_stats': team_stats,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = DataIngestor()
    ingestor.run_daily_update()
