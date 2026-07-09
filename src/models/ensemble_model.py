"""
ensemble_model.py — MLB Ensemble v2.0
XGBoost + RandomForest + GradientBoosting con estrategia de filtrado
"""
import os
import numpy as np
import pandas as pd
import logging
import joblib

logger = logging.getLogger(__name__)

MODEL_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
    'models',
    '../models',
    '../../models',
]

EXPECTED_FEATURES = [
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


def _find_model_dir():
    for d in MODEL_DIRS:
        d = os.path.normpath(d)
        if os.path.isdir(d) and any(f.endswith('.joblib') for f in os.listdir(d)):
            return d
    return None


class MLBEnsembleModel:
    """
    Ensemble de 3 modelos calibrados: XGBoost + RandomForest + GradientBoosting.
    Estrategia de pick: solo emite predicción cuando todos los modelos coinciden.
    Accuracy esperado en picks filtrados: 57-63% (MLB techo real con datos de equipo).
    """

    def __init__(self, model_paths=None):
        self.models = []
        self.scaler = None
        self.feature_names = EXPECTED_FEATURES
        self.logger = logging.getLogger(__name__)

        model_dir = _find_model_dir()

        if model_paths:
            for p in model_paths:
                try:
                    self.models.append(joblib.load(p))
                    self.logger.info(f"✅ Modelo cargado: {p}")
                except Exception as e:
                    self.logger.warning(f"⚠️ No se pudo cargar {p}: {e}")
        elif model_dir:
            for fname in ['xgboost_v1.joblib', 'random_forest_v1.joblib', 'gradient_boosting_v1.joblib']:
                fpath = os.path.join(model_dir, fname)
                if os.path.exists(fpath):
                    try:
                        self.models.append(joblib.load(fpath))
                        self.logger.info(f"✅ {fname} cargado")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error cargando {fname}: {e}")

            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            feat_path = os.path.join(model_dir, 'feature_names.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            if os.path.exists(feat_path):
                self.feature_names = joblib.load(feat_path)

        if not self.models:
            self.logger.warning("⚠️ Sin modelos cargados. Usando heurística determinista.")

    def _prepare_features(self, X):
        """Selecciona y limpia features del DataFrame."""
        if isinstance(X, pd.DataFrame):
            available = [f for f in self.feature_names if f in X.columns]
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                self.logger.warning(f"⚠️ Features faltantes: {missing[:5]}...")
            Xf = X[available].copy() if available else pd.DataFrame(index=X.index)
            # Añadir columnas faltantes con 0
            for f in self.feature_names:
                if f not in Xf.columns:
                    Xf[f] = 0.0
            Xf = Xf[self.feature_names].fillna(0)
        else:
            Xf = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]] if hasattr(X, 'shape') else self.feature_names)
            Xf = Xf.fillna(0)

        if self.scaler is not None:
            Xf = pd.DataFrame(self.scaler.transform(Xf), columns=Xf.columns, index=Xf.index if hasattr(Xf, 'index') else None)
        return Xf

    def predict(self, X):
        """Retorna probabilidades de victoria del equipo local."""
        if not self.models:
            return self._heuristic_predict(X)
        Xf = self._prepare_features(X)
        probs = np.array([m.predict_proba(Xf)[:, 1] for m in self.models])
        return np.mean(probs, axis=0)

    def predict_with_confidence(self, X, threshold=0.57):
        """
        Retorna solo picks donde todos los modelos coinciden en dirección
        y la probabilidad supera el threshold.
        """
        if not self.models:
            probs = self._heuristic_predict(X)
            mask = (probs > threshold) | (probs < (1 - threshold))
            return probs, mask

        Xf = self._prepare_features(X)
        indiv = np.array([m.predict_proba(Xf)[:, 1] for m in self.models])
        ensemble = np.mean(indiv, axis=0)

        # Todos los modelos coinciden en HOME o todos en AWAY
        directions = (indiv > 0.5)
        all_agree = np.all(directions == directions[0], axis=0)

        # Y la probabilidad ensemble supera el threshold
        confident = all_agree & ((ensemble > threshold) | (ensemble < (1 - threshold)))

        return ensemble, confident

    def _heuristic_predict(self, X):
        """
        Heurística determinista basada en diferencial de win%.
        Home advantage base: 54%. Ajusta según win_pct_diff_L10.
        """
        if isinstance(X, pd.DataFrame) and 'home_win_pct_L10' in X.columns and 'away_win_pct_L10' in X.columns:
            home_wp = X['home_win_pct_L10'].fillna(0.5).values
            away_wp = X['away_win_pct_L10'].fillna(0.5).values
            diff = home_wp - away_wp
            # Escalar: diff de 0.5 → probabilidad de 0.70
            probs = 0.54 + diff * 0.4
            return np.clip(probs, 0.30, 0.75)
        else:
            n = len(X) if hasattr(X, '__len__') else 1
            return np.full(n, 0.54)
