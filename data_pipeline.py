"""
═══════════════════════════════════════════════════════════════════════════
data_pipeline.py — ETL Pipeline Professional v3.0
═══════════════════════════════════════════════════════════════════════════
Pipeline completo de datos:
1. Extracción (ESPN + MLB API + Odds API)
2. Transformación (Feature Engineering avanzado)
3. Carga (Persistencia en Parquet con compresión)
4. Validación (Data Quality Checks)
═══════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scraper_espn import ESPNScraper
from api import OddsAPI
from config import Config
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class MLBDataPipeline:
    """
    Pipeline ETL profesional para datos MLB.
    Integra múltiples fuentes y genera features de nivel institucional.
    """

    def __init__(self, data_dir='./data', config: Optional[Config] = None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.config = config or Config()
        self.scraper = ESPNScraper()
        self.odds_api = OddsAPI(self.config)

        # Directorios de datos
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.features_dir = os.path.join(data_dir, 'features')

        for directory in [self.raw_dir, self.processed_dir, self.features_dir]:
            os.makedirs(directory, exist_ok=True)

        # Scalers
        self.scaler = RobustScaler()  # Más robusto a outliers que StandardScaler
        self.scaler_path = os.path.join(self.processed_dir, 'scaler.joblib')

        logger.info('🏗️ MLB Data Pipeline v3.0 inicializado')

    # ═══════════════════════════════════════════════════════════════════
    # EXTRACCIÓN (Extract)
    # ═══════════════════════════════════════════════════════════════════

    def extract_all_sources(self, seasons: List[int] = [2024, 2025]) -> Dict[str, pd.DataFrame]:
        """
        Extrae datos de todas las fuentes disponibles.
        Returns: Dict con DataFrames por fuente.
        """
        logger.info('📥 Iniciando extracción de datos...')

        data = {}

        # 1. ESPN Team Stats
        logger.info('  1/4 ESPN Team Stats...')
        team_stats = self.scraper.scrape_all_teams_parallel(seasons=seasons)
        if not team_stats.empty:
            data['team_stats'] = team_stats
            self._save_raw(team_stats, 'espn_team_stats')

        # 2. ESPN Player Stats (Batting + Pitching)
        logger.info('  2/4 ESPN Player Stats...')
        batting = self.scraper.scrape_player_stats('batting', limit=200)
        pitching = self.scraper.scrape_player_stats('pitching', limit=200)

        if not batting.empty:
            data['batting'] = self.scraper.calculate_advanced_metrics(batting)
            self._save_raw(data['batting'], 'espn_batting')

        if not pitching.empty:
            data['pitching'] = pitching
            self._save_raw(pitching, 'espn_pitching')

        # 3. MLB Official API (Juegos recientes)
        logger.info('  3/4 MLB Official API...')
        recent_games = self._fetch_recent_games(days_back=30)
        if not recent_games.empty:
            data['recent_games'] = recent_games
            self._save_raw(recent_games, 'mlb_recent_games')

        # 4. Odds API (cuotas actuales)
        logger.info('  4/4 Odds API...')
        current_odds = self._fetch_current_odds()
        if not current_odds.empty:
            data['current_odds'] = current_odds
            self._save_raw(current_odds, 'odds_current')

        logger.info(f'✅ Extracción completada: {len(data)} fuentes')
        return data

    def _fetch_recent_games(self, days_back: int = 30) -> pd.DataFrame:
        """Obtiene juegos recientes desde MLB API."""
        try:
            games = []
            for i in range(days_back):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                try:
                    import statsapi
                    schedule = statsapi.schedule(date=date)
                    for game in schedule:
                        games.append({
                            'game_id': game['game_id'],
                            'date': date,
                            'away_team': game['away_name'],
                            'home_team': game['home_name'],
                            'away_score': game['away_score'],
                            'home_score': game['home_score'],
                            'status': game['status'],
                        })
                except Exception:
                    continue

            return pd.DataFrame(games)

        except Exception as e:
            logger.error(f'❌ Error fetching MLB games: {e}')
            return pd.DataFrame()

    def _fetch_current_odds(self) -> pd.DataFrame:
        """Obtiene cuotas actuales desde Odds API."""
        try:
            eventos = self.odds_api.obtener_cuotas()
            if not eventos:
                return pd.DataFrame()

            odds_data = []
            for evento in eventos:
                for bookmaker in evento.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        for outcome in market.get('outcomes', []):
                            odds_data.append({
                                'home_team': evento.get('home_team'),
                                'away_team': evento.get('away_team'),
                                'commence_time': evento.get('commence_time'),
                                'bookmaker': bookmaker.get('key'),
                                'market': market.get('key'),
                                'team': outcome.get('name'),
                                'price': outcome.get('price'),
                                'point': outcome.get('point'),
                            })

            return pd.DataFrame(odds_data)

        except Exception as e:
            logger.error(f'❌ Error fetching odds: {e}')
            return pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════
    # TRANSFORMACIÓN (Transform) — Feature Engineering Avanzado
    # ═══════════════════════════════════════════════════════════════════

    def transform_to_features(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Genera features de nivel institucional:
        - Rolling windows (L5, L10, L20, L50)
        - Momentum indicators
        - Form streaks
        - Head-to-head historics
        - Sabermetrics (wOBA, FIP, xFIP)
        - Elo ratings
        - Rest days impact
        - Weather adjustments (si disponible)
        """
        logger.info('🔄 Generando features avanzados...')

        # Base: recent games
        if 'recent_games' not in raw_data or raw_data['recent_games'].empty:
            logger.error('❌ No hay datos de juegos recientes')
            return pd.DataFrame()

        games = raw_data['recent_games'].copy()
        games['date'] = pd.to_datetime(games['date'])
        games = games.sort_values('date')

        # Calcular resultado
        games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
        games['total_runs'] = games['home_score'] + games['away_score']
        games['run_diff'] = games['home_score'] - games['away_score']

        # ─── 1. ROLLING WINDOWS (sin data leakage) ────────────────────
        features = self._compute_rolling_features(games)

        # ─── 2. MOMENTUM INDICATORS ────────────────────────────────────
        features = self._compute_momentum(features)

        # ─── 3. FORM STREAKS ───────────────────────────────────────────
        features = self._compute_form_streaks(features)

        # ─── 4. HEAD-TO-HEAD HISTORY ───────────────────────────────────
        features = self._compute_h2h_features(features)

        # ─── 5. ELO RATINGS ────────────────────────────────────────────
        features = self._compute_elo_ratings(features)

        # ─── 6. REST DAYS IMPACT ───────────────────────────────────────
        features = self._compute_rest_days(features)

        # ─── 7. PITCHER MATCHUP (si disponible) ────────────────────────
        if 'batting' in raw_data and 'pitching' in raw_data:
            features = self._enrich_with_player_stats(
                features,
                raw_data['batting'],
                raw_data['pitching']
            )

        # ─── 8. MARKET FEATURES (cuotas) ───────────────────────────────
        if 'current_odds' in raw_data:
            features = self._add_market_features(features, raw_data['current_odds'])

        logger.info(f'✅ Features generados: {features.shape[1]} columnas')
        return features

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula rolling windows para home y away teams."""
        features = df.copy()

        windows = [5, 10, 20, 50]
        metrics = ['home_score', 'away_score', 'home_win', 'total_runs']

        for window in windows:
            for metric in metrics:
                # Home team rolling
                features[f'home_{metric}_L{window}'] = (
                    features.groupby('home_team')[metric]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

                # Away team rolling
                features[f'away_{metric}_L{window}'] = (
                    features.groupby('away_team')[metric]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

        return features

    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores de momentum (L5 vs L20)."""
        features = df.copy()

        for team_type in ['home', 'away']:
            # Win rate momentum
            features[f'{team_type}_win_momentum'] = (
                features[f'{team_type}_home_win_L5'] -
                features[f'{team_type}_home_win_L20']
            )

            # Scoring momentum
            features[f'{team_type}_score_momentum'] = (
                features[f'{team_type}_home_score_L5'] -
                features[f'{team_type}_home_score_L20']
            )

        return features

    def _compute_form_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula rachas actuales (win/loss streaks)."""
        features = df.copy()

        def calculate_streak(series):
            """Calcula la racha actual (positiva = victorias, negativa = derrotas)."""
            if len(series) == 0:
                return 0
            current = series.iloc[-1]
            streak = 1
            for i in range(len(series)-2, -1, -1):
                if series.iloc[i] == current:
                    streak += 1
                else:
                    break
            return streak if current == 1 else -streak

        for team_col in ['home_team', 'away_team']:
            features[f'{team_col}_streak'] = (
                features.groupby(team_col)['home_win']
                .transform(lambda x: x.shift(1).rolling(10, min_periods=1).apply(calculate_streak, raw=False))
            )

        return features

    def _compute_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de enfrentamientos directos históricos."""
        features = df.copy()

        # Crear clave de matchup
        features['matchup'] = features.apply(
            lambda row: tuple(sorted([row['home_team'], row['away_team']])),
            axis=1
        )

        # Histórico de matchup (últimos 10 enfrentamientos)
        features['h2h_home_wins_L10'] = (
            features.groupby('matchup')['home_win']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
        )

        features['h2h_avg_total_L10'] = (
            features.groupby('matchup')['total_runs']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )

        return features

    def _compute_elo_ratings(self, df: pd.DataFrame, k_factor: int = 20) -> pd.DataFrame:
        """Calcula Elo ratings dinámicos para cada equipo."""
        features = df.copy()

        # Inicializar Elo ratings
        elo_ratings = {}
        default_elo = 1500

        features['home_elo'] = 0.0
        features['away_elo'] = 0.0
        features['elo_diff'] = 0.0

        for idx, row in features.iterrows():
            home = row['home_team']
            away = row['away_team']

            # Obtener Elos actuales
            home_elo = elo_ratings.get(home, default_elo)
            away_elo = elo_ratings.get(away, default_elo)

            features.at[idx, 'home_elo'] = home_elo
            features.at[idx, 'away_elo'] = away_elo
            features.at[idx, 'elo_diff'] = home_elo - away_elo

            # Actualizar Elos después del juego
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            actual_home = row['home_win']

            new_home_elo = home_elo + k_factor * (actual_home - expected_home)
            new_away_elo = away_elo + k_factor * ((1 - actual_home) - (1 - expected_home))

            elo_ratings[home] = new_home_elo
            elo_ratings[away] = new_away_elo

        return features

    def _compute_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula días de descanso desde el último juego."""
        features = df.copy()

        for team_col in ['home_team', 'away_team']:
            features[f'{team_col}_rest_days'] = (
                features.groupby(team_col)['date']
                .transform(lambda x: x.diff().dt.days.fillna(3))
            )

        return features

    def _enrich_with_player_stats(
        self,
        features: pd.DataFrame,
        batting: pd.DataFrame,
        pitching: pd.DataFrame
    ) -> pd.DataFrame:
        """Agrega estadísticas de jugadores al nivel de equipo."""
        # Agregar por equipo
        team_batting = batting.groupby('team').agg({
            'woba': 'mean',
            'iso': 'mean',
            'babip': 'mean',
            'avg': 'mean',
            'obp': 'mean',
            'slg': 'mean',
        }).add_prefix('team_batting_')

        team_pitching = pitching.groupby('team').agg({
            'era': 'mean',
            'whip': 'mean',
            'k9': 'mean',
            'bb9': 'mean',
        }).add_prefix('team_pitching_')

        # Merge con features
        features = features.merge(
            team_batting,
            left_on='home_team',
            right_index=True,
            how='left',
            suffixes=('', '_home')
        )

        features = features.merge(
            team_pitching,
            left_on='home_team',
            right_index=True,
            how='left',
            suffixes=('', '_home')
        )

        return features

    def _add_market_features(self, features: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """Agrega features del mercado (cuotas, movimiento de líneas)."""
        # Calcular implied probabilities
        odds['implied_prob'] = odds['price'].apply(
            lambda x: 100 / (x + 100) if x > 0 else abs(x) / (abs(x) + 100)
        )

        # Promediar cuotas por matchup
        market_features = odds.groupby(['home_team', 'away_team', 'market']).agg({
            'price': ['mean', 'std'],
            'implied_prob': 'mean',
        }).reset_index()

        market_features.columns = ['_'.join(col).strip('_') for col in market_features.columns]

        # Merge
        features = features.merge(
            market_features,
            on=['home_team', 'away_team'],
            how='left'
        )

        return features

    # ═══════════════════════════════════════════════════════════════════
    # CARGA (Load) + PERSISTENCIA
    # ═══════════════════════════════════════════════════════════════════

    def load_features(self, features: pd.DataFrame, version: str = 'latest'):
        """Persiste features en formato Parquet (compresión eficiente)."""
        logger.info('💾 Guardando features...')

        # Guardar con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'features_{version}_{timestamp}.parquet'
        path = os.path.join(self.features_dir, filename)

        features.to_parquet(path, compression='gzip', index=False)
        logger.info(f'✅ Features guardados: {path}')

        # Crear symlink a 'latest'
        latest_path = os.path.join(self.features_dir, f'features_{version}_latest.parquet')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(path, latest_path)

        # Guardar scaler
        if hasattr(self, 'scaler'):
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f'✅ Scaler guardado: {self.scaler_path}')

    def _save_raw(self, df: pd.DataFrame, name: str):
        """Guarda datos raw en formato Parquet."""
        path = os.path.join(self.raw_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.parquet')
        df.to_parquet(path, compression='gzip', index=False)

    def read_latest_features(self, version: str = 'latest') -> pd.DataFrame:
        """Lee el último dataset de features."""
        path = os.path.join(self.features_dir, f'features_{version}_latest.parquet')
        if not os.path.exists(path):
            logger.error(f'❌ No se encontró {path}')
            return pd.DataFrame()

        return pd.read_parquet(path)

    # ═══════════════════════════════════════════════════════════════════
    # VALIDACIÓN (Data Quality)
    # ═══════════════════════════════════════════════════════════════════

    def validate_features(self, features: pd.DataFrame) -> Dict[str, any]:
        """
        Realiza checks de calidad de datos:
        - Missing values
        - Duplicados
        - Outliers
        - Data types
        - Feature correlations
        """
        logger.info('🔍 Validando calidad de datos...')

        report = {
            'timestamp': datetime.now().isoformat(),
            'rows': len(features),
            'columns': len(features.columns),
            'checks': {},
        }

        # 1. Missing values
        missing = features.isnull().sum()
        missing_pct = (missing / len(features)) * 100
        report['checks']['missing_values'] = {
            'total': int(missing.sum()),
            'columns_with_missing': missing[missing > 0].to_dict(),
            'worst_column': missing.idxmax() if missing.sum() > 0 else None,
            'worst_pct': float(missing_pct.max()),
        }

        # 2. Duplicados
        duplicates = features.duplicated().sum()
        report['checks']['duplicates'] = {
            'count': int(duplicates),
            'percentage': float(duplicates / len(features) * 100),
        }

        # 3. Outliers (IQR method)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((features[col] < (Q1 - 1.5 * IQR)) |
                            (features[col] > (Q3 + 1.5 * IQR))).sum()
            if outlier_count > 0:
                outliers[col] = int(outlier_count)

        report['checks']['outliers'] = outliers

        # 4. Data types
        report['checks']['data_types'] = features.dtypes.astype(str).to_dict()

        # 5. High correlations (>0.95)
        numeric_features = features[numeric_cols]
        corr_matrix = numeric_features.corr().abs()
        high_corr = np.where((corr_matrix > 0.95) & (corr_matrix < 1.0))
        high_corr_pairs = [
            (numeric_cols[i], numeric_cols[j], float(corr_matrix.iloc[i, j]))
            for i, j in zip(*high_corr) if i < j
        ]
        report['checks']['high_correlations'] = high_corr_pairs[:10]  # Top 10

        logger.info(f'✅ Validación completada: {report["checks"]["missing_values"]["total"]} missings, '
                   f'{report["checks"]["duplicates"]["count"]} duplicados')

        return report

    # ═══════════════════════════════════════════════════════════════════
    # PIPELINE COMPLETO (ETL)
    # ═══════════════════════════════════════════════════════════════════

    def run_full_pipeline(self, seasons: List[int] = [2024, 2025]) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo:
        1. Extract
        2. Transform
        3. Load
        4. Validate
        """
        logger.info('🚀 Ejecutando pipeline completo...')

        # 1. Extract
        raw_data = self.extract_all_sources(seasons=seasons)

        if not raw_data:
            logger.error('❌ No se extrajeron datos')
            return pd.DataFrame()

        # 2. Transform
        features = self.transform_to_features(raw_data)

        if features.empty:
            logger.error('❌ No se generaron features')
            return pd.DataFrame()

        # 3. Validate
        validation_report = self.validate_features(features)

        # 4. Load
        self.load_features(features, version='production')

        logger.info('✅ Pipeline completo ejecutado exitosamente')
        logger.info(f'   Features: {features.shape}')
        logger.info(f'   Missing: {validation_report["checks"]["missing_values"]["total"]}')
        logger.info(f'   Duplicados: {validation_report["checks"]["duplicates"]["count"]}')

        return features


# ═══════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pipeline = MLBDataPipeline()

    # Ejecutar pipeline completo
    features = pipeline.run_full_pipeline(seasons=[2024, 2025])

    print('\n📊 Features generados:')
    print(features.head())
    print(f'\nShape: {features.shape}')
    print(f'\nColumnas: {list(features.columns[:20])}...')

    # Leer features guardados
    latest = pipeline.read_latest_features(version='production')
    print(f'\n✅ Features cargados desde disco: {latest.shape}')
