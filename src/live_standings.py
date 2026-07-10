"""
live_standings.py — Récord actual de temporada 2026 y probable pitchers
Fuente: statsapi.mlb.com (libre, sin auth)
"""
import requests
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

STANDINGS_CACHE = {}
PITCHERS_CACHE  = {}

# ── 1. RÉCORD ACTUAL DE TEMPORADA ─────────────────────────────────────────────
def fetch_live_standings(season: int = 2026) -> dict:
    """
    Retorna dict {team_name: {'wins': W, 'losses': L, 'win_pct': pct, 'gb': GB}}
    """
    global STANDINGS_CACHE
    cache_key = f"standings_{season}_{datetime.now().strftime('%Y-%m-%d')}"
    if cache_key in STANDINGS_CACHE:
        return STANDINGS_CACHE[cache_key]

    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        standings = {}
        for record in data.get('records', []):
            for team_rec in record.get('teamRecords', []):
                name = team_rec['team']['name']
                W    = team_rec.get('wins', 0)
                L    = team_rec.get('losses', 0)
                pct  = W / (W + L) if (W + L) > 0 else 0.500
                gb   = team_rec.get('gamesBack', '0')
                try:
                    gb = float(str(gb).replace('-','0'))
                except:
                    gb = 0.0
                streak_val = team_rec.get('streak', {}).get('streakNumber', 0)
                streak_dir = team_rec.get('streak', {}).get('streakType', 'W')
                streak     = streak_val if streak_dir == 'W' else -streak_val

                standings[name] = {
                    'wins':     W,
                    'losses':   L,
                    'win_pct':  round(pct, 3),
                    'games':    W + L,
                    'gb':       gb,
                    'streak':   streak,
                }

        STANDINGS_CACHE[cache_key] = standings
        logger.info(f"✅ Standings 2026: {len(standings)} equipos")
        return standings

    except Exception as e:
        logger.warning(f"⚠️ Standings error: {e}")
        return {}


# ── 2. PROBABLE PITCHERS DEL DÍA ──────────────────────────────────────────────
def fetch_probable_pitchers(date_str: str) -> dict:
    """
    Retorna dict {game_id: {'home_pitcher': {...}, 'away_pitcher': {...}}}
    con ERA actual de temporada si está disponible.
    """
    global PITCHERS_CACHE
    if date_str in PITCHERS_CACHE:
        return PITCHERS_CACHE[date_str]

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_str}"
        f"&hydrate=probablePitcher(note),team"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()

        pitchers = {}
        for date_entry in r.json().get('dates', []):
            for game in date_entry.get('games', []):
                gid  = game.get('gamePk')
                home = game['teams']['home']
                away = game['teams']['away']

                def extract_pitcher(side_data):
                    p = side_data.get('probablePitcher', {})
                    if not p:
                        return None
                    return {
                        'id':          p.get('id'),
                        'name':        p.get('fullName', 'TBD'),
                        'era_season':  None,  # se enriquece abajo
                    }

                pitchers[gid] = {
                    'home_pitcher': extract_pitcher(home),
                    'away_pitcher': extract_pitcher(away),
                    'home_team':    home['team']['name'],
                    'away_team':    away['team']['name'],
                }

        # Enriquecer con ERA actual de cada pitcher
        pitcher_ids = set()
        for gid, data in pitchers.items():
            for side in ['home_pitcher', 'away_pitcher']:
                if data[side] and data[side]['id']:
                    pitcher_ids.add(data[side]['id'])

        era_map = _fetch_pitcher_era_current(list(pitcher_ids))

        for gid, data in pitchers.items():
            for side in ['home_pitcher', 'away_pitcher']:
                if data[side] and data[side]['id']:
                    pid = data[side]['id']
                    data[side]['era_season'] = era_map.get(pid, 4.50)

        PITCHERS_CACHE[date_str] = pitchers
        logger.info(f"✅ Probable pitchers: {len(pitchers)} juegos | ERA enriquecida para {len(era_map)} pitchers")
        return pitchers

    except Exception as e:
        logger.warning(f"⚠️ Probable pitchers error: {e}")
        return {}


def _fetch_pitcher_era_current(pitcher_ids: list) -> dict:
    """ERA actual de temporada 2026 para una lista de pitcher IDs."""
    era_map = {}
    for pid in pitcher_ids[:40]:   # Limitar para evitar rate-limit
        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season&group=pitching&season=2026"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            splits = r.json().get('stats', [{}])[0].get('splits', [])
            if splits:
                stat = splits[0].get('stat', {})
                era  = stat.get('era', '4.50')
                try:
                    era_map[pid] = float(era)
                except:
                    era_map[pid] = 4.50
        except:
            continue
    return era_map


# ── 3. ENRIQUECER DATAFRAME DE JUEGOS ─────────────────────────────────────────
def enrich_with_live_data(df_games: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Añade al dataframe de juegos:
    - home_season_wins, home_season_losses, home_season_win_pct
    - away_season_wins, away_season_losses, away_season_win_pct
    - home_season_streak, away_season_streak
    - home_gb, away_gb (games behind líder de división)
    - home_pitcher_era, away_pitcher_era (ERA real de temporada)
    - home_pitcher_name, away_pitcher_name
    """
    df = df_games.copy()

    # Standings
    standings = fetch_live_standings(2026)

    LEAGUE_AVG_ERA = 4.30   # 2026 promedio MLB

    for side in ['home', 'away']:
        df[f'{side}_season_wins']    = df[f'{side}_team'].map(lambda t: standings.get(t,{}).get('wins',    0))
        df[f'{side}_season_losses']  = df[f'{side}_team'].map(lambda t: standings.get(t,{}).get('losses',  0))
        df[f'{side}_season_win_pct'] = df[f'{side}_team'].map(lambda t: standings.get(t,{}).get('win_pct', 0.500))
        df[f'{side}_season_streak']  = df[f'{side}_team'].map(lambda t: standings.get(t,{}).get('streak',  0))
        df[f'{side}_gb']             = df[f'{side}_team'].map(lambda t: standings.get(t,{}).get('gb',      0.0))
        df[f'{side}_pitcher_era']    = LEAGUE_AVG_ERA
        df[f'{side}_pitcher_name']   = 'TBD'

    # Win% diferencial (temporada completa — el feature más potente)
    df['season_win_pct_diff'] = df['home_season_win_pct'] - df['away_season_win_pct']
    df['season_wins_diff']    = df['home_season_wins']    - df['away_season_wins']
    df['streak_diff']         = df['home_season_streak']  - df['away_season_streak']

    # Probable pitchers
    pitchers = fetch_probable_pitchers(date_str)

    for idx, row in df.iterrows():
        gid = row.get('game_id')
        if gid and int(gid) in pitchers:
            p_data = pitchers[int(gid)]
            for side in ['home', 'away']:
                p = p_data.get(f'{side}_pitcher')
                if p:
                    df.at[idx, f'{side}_pitcher_era']  = p.get('era_season', LEAGUE_AVG_ERA)
                    df.at[idx, f'{side}_pitcher_name']  = p.get('name', 'TBD')

    # ERA diferencial de starters (ajustado a temporada actual)
    df['pitcher_era_diff'] = df['away_pitcher_era'] - df['home_pitcher_era']

    logger.info(
        f"✅ Live data integrado | "
        f"standings {len(standings)} equipos | "
        f"pitchers {len(pitchers)} juegos"
    )
    return df
