"""
═══════════════════════════════════════════════════════════════════════════
model.py — Ensemble ML con Rolling Windows + Time Decay
═══════════════════════════════════════════════════════════════════════════
- XGBoost + LightGBM + RandomForest (Ensemble real)
- Ventanas móviles: L5, L10, L25 (sin data leakage)
- Time Decay: 30% más peso a últimas 2 semanas
- Mercados: Moneyline, Totals, Handicaps
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
import logging
import os
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ─── ML Imports ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        VotingClassifier, VotingRegressor,
        GradientBoostingClassifier
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MLBQuantModel:
    """
    Motor de predicción cuantitativo para MLB.
    Ensemble: XGBoost + LightGBM + RandomForest
    Con Rolling Windows y Time Decay para prevenir data leakage.
    """

    def __init__(self):
        self.home_advantage = 0.035  # MLB home advantage ~3.5%
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Ensemble models
        self.ml_classifier = None  # Para moneyline (clasificación)
        self.total_regressor = None  # Para totals (regresión)
        self.spread_regressor = None  # Para spreads (regresión)
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False

        # Time decay config
        self.decay_half_life_days = 14  # 2 semanas
        self.decay_boost = 0.30  # 30% más peso

        # Feature columns
        self.feature_cols = None

        # Intentar cargar modelos pre-entrenados
        self._load_models()

        logger.info('🤖 MLBQuantModel v2.0 — Ensemble + Rolling Windows + Time Decay')

    # ═══════════════════════════════════════════════════════════════════
    # ROLLING WINDOWS (prevención de data leakage)
    # ═══════════════════════════════════════════════════════════════════

    def compute_rolling_features(self, df, game_date_col='game_date', team_col='team'):
        """
        Calcula features con ventanas móviles (L5, L10, L25).
        SOLO usa datos ANTERIORES a cada fecha (sin data leakage).
        """
        logger.info("📊 Calculando rolling features (anti-leakage)...")

        df = df.sort_values(game_date_col).copy()
        df[game_date_col] = pd.to_datetime(df[game_date_col])

        # Métricas base para rolling
        numeric_cols = [
            'runs_scored', 'runs_allowed', 'hits', 'errors',
            'strikeouts', 'walks', 'home_runs', 'win'
        ]
        available_numeric = [c for c in numeric_cols if c in df.columns]

        for window in [5, 10, 25]:
            for col in available_numeric:
                # shift(1) asegura que NO incluimos el juego actual
                df[f'{col}_L{window}'] = (
                    df.groupby(team_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

        # Tendencia: diferencia entre L5 y L25 (momentum)
        for col in available_numeric:
            if f'{col}_L5' in df.columns and f'{col}_L25' in df.columns:
                df[f'{col}_momentum'] = df[f'{col}_L5'] - df[f'{col}_L25']

        return df

    # ═══════════════════════════════════════════════════════════════════
    # TIME DECAY (30% más peso a últimas 2 semanas)
    # ═══════════════════════════════════════════════════════════════════

    def compute_sample_weights(self, dates, reference_date=None):
        """
        Calcula pesos de muestra con decaimiento temporal.
        Partidos de últimas 2 semanas reciben 30% más peso.
        """
        if reference_date is None:
            reference_date = datetime.now()

        dates = pd.to_datetime(dates)
        days_ago = (reference_date - dates).dt.days

        # Exponential decay
        weights = np.exp(-0.5 * days_ago / self.decay_half_life_days)

        # Boost adicional para últimas 2 semanas
        recent_mask = days_ago <= 14
        weights[recent_mask] *= (1 + self.decay_boost)

        # Normalizar para que sumen a len(weights)
        weights = weights / weights.mean()

        return weights.values

    # ═══════════════════════════════════════════════════════════════════
    # ENTRENAMIENTO DEL ENSEMBLE
    # ═══════════════════════════════════════════════════════════════════

    def train(self, training_data, target_col='result', date_col='game_date'):
        """
        Entrena el ensemble con validación temporal (TimeSeriesSplit).
        training_data: DataFrame con features + target + game_date
        """
        if not SKLEARN_AVAILABLE:
            logger.error("❌ scikit-learn no disponible")
            return

        logger.info("🔄 Entrenando Ensemble ML...")

        df = training_data.sort_values(date_col).copy()

        # Separar features
        exclude_cols = [target_col, date_col, 'team', 'opponent', 'game_id']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols
                            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        X = df[self.feature_cols].fillna(0)
        y = df[target_col]

        # Sample weights con time decay
        weights = self.compute_sample_weights(df[date_col])

        # Escalar features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_cols,
            index=X.index
        )

        # ─── Validación Temporal (TimeSeriesSplit) ────────────────────
        tscv = TimeSeriesSplit(n_splits=5)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_train = weights[train_idx]

            # Entrenar modelos individuales
            models = self._build_classifiers()
            for name, model in models.items():
                try:
                    if hasattr(model, 'sample_weight'):
                        model.fit(X_train, y_train, sample_weight=w_train)
                    else:
                        model.fit(X_train, y_train)
                except Exception:
                    model.fit(X_train, y_train)

            # Evaluar fold
            fold_preds = self._ensemble_predict_proba(models, X_val)
            if fold_preds is not None and len(np.unique(y_val)) > 1:
                score = log_loss(y_val, fold_preds, labels=[0, 1])
                val_scores.append(score)
                logger.info(f"  Fold {fold+1}: LogLoss={score:.4f}")

        # ─── Entrenamiento final con TODOS los datos ──────────────────
        self.classifiers = self._build_classifiers()
        for name, model in self.classifiers.items():
            try:
                model.fit(X_scaled, y, sample_weight=weights)
            except Exception:
                model.fit(X_scaled, y)

        # Calibrar probabilidades (Platt Scaling)
        self.calibrated_classifiers = {}
        for name, model in self.classifiers.items():
            try:
                cal = CalibratedClassifierCV(model, cv=3, method='isotonic')
                cal.fit(X_scaled, y, sample_weight=weights)
                self.calibrated_classifiers[name] = cal
            except Exception:
                self.calibrated_classifiers[name] = model

        # ─── Total Regressor ─────────────────────────────────────────
        if 'total_runs' in df.columns:
            y_total = df['total_runs']
            self.total_regressor = self._build_regressors()
            for name, model in self.total_regressor.items():
                model.fit(X_scaled, y_total, sample_weight=weights)

        # ─── Spread Regressor ────────────────────────────────────────
        if 'run_diff' in df.columns:
            y_spread = df['run_diff']
            self.spread_regressor = self._build_regressors()
            for name, model in self.spread_regressor.items():
                model.fit(X_scaled, y_spread, sample_weight=weights)

        self.is_trained = True
        self._save_models()

        avg_score = np.mean(val_scores) if val_scores else 0
        logger.info(f"✅ Ensemble entrenado | Avg LogLoss: {avg_score:.4f}")
        logger.info(f"   Features: {len(self.feature_cols)} | Samples: {len(X)}")

    def _build_classifiers(self):
        """Construye los clasificadores del ensemble."""
        models = {}

        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
            )

        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbosity=-1,
            )

        if SKLEARN_AVAILABLE:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )

        return models

    def _build_regressors(self):
        """Construye los regresores para totals/spreads."""
        models = {}

        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, verbosity=0,
            )

        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, verbosity=-1,
            )

        if SKLEARN_AVAILABLE:
            models['random_forest'] = RandomForestRegressor(
                n_estimators=150, max_depth=8, n_jobs=-1, random_state=42,
            )

        return models

    def _ensemble_predict_proba(self, models, X):
        """Promedio ponderado de probabilidades del ensemble."""
        # Pesos: XGBoost > LightGBM > RF
        model_weights = {'xgboost': 0.40, 'lightgbm': 0.35, 'random_forest': 0.25}
        predictions = []
        weights = []

        for name, model in models.items():
            try:
                pred = model.predict_proba(X)
                predictions.append(pred)
                weights.append(model_weights.get(name, 0.33))
            except Exception:
                continue

        if not predictions:
            return None

        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        return ensemble_pred

    # ═══════════════════════════════════════════════════════════════════
    # PREDICCIONES
    # ═══════════════════════════════════════════════════════════════════

    def predict_moneyline(self, home_team, away_team, odds_price, features=None):
        """
        Predice probabilidad de victoria del home team.
        Si el modelo está entrenado, usa el ensemble.
        Si no, usa método heurístico mejorado con implied odds.
        """
        if self.is_trained and features is not None and self.feature_cols:
            return self._predict_ml(features)

        # Fallback: implied probability + home advantage + adjustments
        if odds_price > 0:
            implied_prob = 100 / (odds_price + 100)
        else:
            implied_prob = abs(odds_price) / (abs(odds_price) + 100)

        # Ajuste por ventaja local
        adjusted = implied_prob + self.home_advantage

        # Si tenemos features de rolling, ajustar
        if features:
            momentum_adj = self._momentum_adjustment(features)
            adjusted += momentum_adj

        return max(0.15, min(0.85, adjusted))

    def _predict_ml(self, features):
        """Predicción usando el ensemble ML entrenado."""
        try:
            X = pd.DataFrame([features])[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)

            predictions = []
            weights = []
            model_weights = {'xgboost': 0.40, 'lightgbm': 0.35, 'random_forest': 0.25}

            for name, model in self.calibrated_classifiers.items():
                try:
                    pred = model.predict_proba(X_scaled)[0][1]  # P(home_win)
                    predictions.append(pred)
                    weights.append(model_weights.get(name, 0.33))
                except Exception:
                    continue

            if predictions:
                weights = np.array(weights) / sum(weights)
                return float(np.average(predictions, weights=weights))

        except Exception as e:
            logger.warning(f"⚠️ ML prediction failed: {e}")

        return None

    def _momentum_adjustment(self, features):
        """Ajusta la predicción basado en momentum (L5 vs L25)."""
        adj = 0.0
        # Win rate momentum
        home_momentum = features.get('home_win_pct_L5', 0.5) - features.get('home_win_pct_L25', 0.5)
        away_momentum = features.get('away_win_pct_L5', 0.5) - features.get('away_win_pct_L25', 0.5)
        adj += (home_momentum - away_momentum) * 0.10

        # Runs scored momentum
        home_rs_mom = features.get('home_runs_scored_L5', 4.5) - features.get('home_runs_scored_L25', 4.5)
        away_rs_mom = features.get('away_runs_scored_L5', 4.5) - features.get('away_runs_scored_L25', 4.5)
        adj += (home_rs_mom - away_rs_mom) * 0.02

        return np.clip(adj, -0.08, 0.08)

    def predict_total(self, home_team, away_team, features=None):
        """
        Predice total de carreras.
        Si el modelo está entrenado, usa ensemble regressor.
        """
        if self.is_trained and self.total_regressor and features:
            try:
                X = pd.DataFrame([features])[self.feature_cols].fillna(0)
                X_scaled = self.scaler.transform(X)

                predictions = []
                for name, model in self.total_regressor.items():
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)

                return float(np.mean(predictions))
            except Exception:
                pass

        # Fallback heurístico mejorado
        base_total = 8.5
        if features:
            # Ajustar por runs scored recientes de ambos equipos
            home_rs = features.get('home_runs_scored_L10', 4.5)
            away_rs = features.get('away_runs_scored_L10', 4.5)
            home_ra = features.get('home_runs_allowed_L10', 4.5)
            away_ra = features.get('away_runs_allowed_L10', 4.5)

            # Estimación: (home_offense + away_defense) / 2 + (away_offense + home_defense) / 2
            estimated = ((home_rs + away_ra) / 2) + ((away_rs + home_ra) / 2)
            base_total = estimated * 0.6 + base_total * 0.4

        return round(base_total, 1)

    def predict_spread(self, home_team, away_team, team, features=None):
        """
        Predice el spread (diferencia de carreras).
        Positivo = home gana por X, Negativo = away gana por X.
        """
        if self.is_trained and self.spread_regressor and features:
            try:
                X = pd.DataFrame([features])[self.feature_cols].fillna(0)
                X_scaled = self.scaler.transform(X)

                predictions = []
                for name, model in self.spread_regressor.items():
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)

                spread = float(np.mean(predictions))
                return spread if team == home_team else -spread
            except Exception:
                pass

        # Fallback heurístico
        base_spread = 0.5 if team == home_team else -0.5
        if features:
            home_diff = features.get('home_runs_scored_L10', 4.5) - features.get('home_runs_allowed_L10', 4.5)
            away_diff = features.get('away_runs_scored_L10', 4.5) - features.get('away_runs_allowed_L10', 4.5)
            edge = (home_diff - away_diff) * 0.3
            base_spread += edge if team == home_team else -edge

        return round(np.clip(base_spread, -3.0, 3.0), 1)

    # ═══════════════════════════════════════════════════════════════════
    # EDGE CALCULATION (Kelly Criterion)
    # ═══════════════════════════════════════════════════════════════════

    def calculate_edge(self, model_prob, market_odds):
        """
        Calcula el edge y Kelly fraccional.
        market_odds: American odds format
        """
        if market_odds > 0:
            implied_prob = 100 / (market_odds + 100)
            decimal_odds = (market_odds / 100) + 1
        else:
            implied_prob = abs(market_odds) / (abs(market_odds) + 100)
            decimal_odds = (100 / abs(market_odds)) + 1

        # Edge = model_prob - implied_prob
        edge = (model_prob - implied_prob) * 100  # En porcentaje

        # Kelly Criterion (fraccional 0.25)
        kelly_full = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
        kelly_fraction = max(0, kelly_full * 0.25)

        # Expected Value
        ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)

        return {
            'edge_pct': round(edge, 2),
            'kelly_fraction': round(kelly_fraction, 4),
            'ev': round(ev, 4),
            'implied_prob': round(implied_prob, 4),
            'model_prob': round(model_prob, 4),
            'decimal_odds': round(decimal_odds, 3),
        }

    # ═══════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════

    def _save_models(self):
        """Guarda modelos entrenados."""
        try:
            save_data = {
                'classifiers': self.classifiers,
                'calibrated': self.calibrated_classifiers,
                'total_regressor': self.total_regressor,
                'spread_regressor': self.spread_regressor,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'is_trained': self.is_trained,
            }
            path = os.path.join(self.models_dir, 'ensemble_mlb.joblib')
            joblib.dump(save_data, path)
            logger.info(f"💾 Modelos guardados en {path}")
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")

    def _load_models(self):
        """Carga modelos pre-entrenados."""
        path = os.path.join(self.models_dir, 'ensemble_mlb.joblib')
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                self.classifiers = data.get('classifiers', {})
                self.calibrated_classifiers = data.get('calibrated', {})
                self.total_regressor = data.get('total_regressor')
                self.spread_regressor = data.get('spread_regressor')
                self.scaler = data.get('scaler', self.scaler)
                self.feature_cols = data.get('feature_cols')
                self.is_trained = data.get('is_trained', False)
                if self.is_trained:
                    logger.info("✅ Modelos pre-entrenados cargados")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando modelos: {e}")

    def retrain(self, training_data):
        """Reentrena el ensemble con nuevos datos."""
        logger.info('🔄 Reentrenando ensemble...')
        # Primero calcular rolling features
        df = self.compute_rolling_features(training_data)
        self.train(df)
        logger.info('✅ Ensemble reentrenado.')
