"""
mlb_ensemble_v3.py — MLB Prediction Engine v3.0
Poisson + XGBoost + Decision Tree + Logistic Regression
Diseñado para maximizar acierto en spots selectivos (edge > 5%).
Usa rolling windows L5/L10/L25, time decay, y sabermetría.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import poisson
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

try:
    from xgboost import XGBClassifier
    XGBOOST = True
except ImportError:
    XGBOOST = False

import joblib
import os
from datetime import datetime, timedelta

logger = __import__('logging').getLogger(__name__)


# ── 1. FEATURE ENGINEERING ────────────────────────────────────────────────────

class FeatureEngineer:
    """Rolling windows anti-leakage con time decay."""

    def __init__(self):
        self.windows = [5, 10, 25]
        self.decay_half_life = 14
        self.decay_boost = 0.30

    def build_features(self, games_df):
        df = games_df.sort_values('date').copy()
        df['date'] = pd.to_datetime(df['date'])
        df['home_win'] = (df['home_runs'] > df['away_runs']).astype(int)
        df['total_runs'] = df['home_runs'] + df['away_runs']
        df['run_diff'] = df['home_runs'] - df['away_runs']

        all_features = []
        for idx, row in df.iterrows():
            home, away = row['home_team'], row['away_team']
            game_date = row['date']

            home_prev = df[
                ((df['home_team'] == home) | (df['away_team'] == home)) &
                (df['date'] < game_date)
            ].tail(25)

            away_prev = df[
                ((df['home_team'] == away) | (df['away_team'] == away)) &
                (df['date'] < game_date)
            ].tail(25)

            features = self._compute_team_rolling(home_prev, home, 'home')
            features.update(self._compute_team_rolling(away_prev, away, 'away'))

            # Starter features — preferir adj_era (park-adjusted) sobre ERA cruda
            era_map = {
                'home_starter_era':  row.get('home_adj_era',  row.get('home_starter_era',  np.nan)),
                'away_starter_era':  row.get('away_adj_era',  row.get('away_starter_era',  np.nan)),
                'home_starter_whip': row.get('home_adj_whip', row.get('home_starter_whip', np.nan)),
                'away_starter_whip': row.get('away_adj_whip', row.get('away_starter_whip', np.nan)),
                'home_starter_k9':   row.get('home_starter_k9', np.nan),
                'away_starter_k9':   row.get('away_starter_k9', np.nan),
                'home_bullpen_era':  row.get('home_adj_bullpen_era', row.get('home_bullpen_era', np.nan)),
                'away_bullpen_era':  row.get('away_adj_bullpen_era', row.get('away_bullpen_era', np.nan)),
                'park_factor':       row.get('park_factor', 1.0),
                'delta_adj_era':     row.get('delta_adj_era', np.nan),
            }
            for col, val in era_map.items():
                features[col] = val

            # Sabermetría avanzada
            for col in ['home_xwoba','away_xwoba','home_barrel_pct','away_barrel_pct',
                        'home_hard_hit_pct','away_hard_hit_pct']:
                features[col] = row.get(col, np.nan)

            # Deltas
            features['delta_era']         = features.get('delta_adj_era', features.get('away_starter_era',4.5) - features.get('home_starter_era',4.5))
            features['delta_whip']        = features.get('away_starter_whip',1.3) - features.get('home_starter_whip',1.3)
            features['delta_k9']          = features.get('home_starter_k9',8.0) - features.get('away_starter_k9',8.0)
            features['delta_bullpen_era'] = features.get('away_bullpen_era',4.0) - features.get('home_bullpen_era',4.0)

            # Momentum
            features['home_momentum']         = features.get('home_win_pct_L5',0.5) - features.get('home_win_pct_L25',0.5)
            features['away_momentum']         = features.get('away_win_pct_L5',0.5) - features.get('away_win_pct_L25',0.5)
            features['momentum_diff']         = features['home_momentum'] - features['away_momentum']

            # ── Temporada actual 2026 (standings en vivo) ─────────────────────
            features['home_season_win_pct']  = row.get('home_season_win_pct', 0.500)
            features['away_season_win_pct']  = row.get('away_season_win_pct', 0.500)
            features['season_win_pct_diff']  = row.get('season_win_pct_diff',
                features['home_season_win_pct'] - features['away_season_win_pct'])
            features['season_wins_diff']     = row.get('season_wins_diff', 0)
            features['home_season_streak']   = row.get('home_season_streak', 0)
            features['away_season_streak']   = row.get('away_season_streak', 0)
            features['streak_diff']          = row.get('streak_diff', 0)

            # ── Pitcher ERA temporada actual (reemplaza ERA histórica) ─────────
            features['home_pitcher_era']     = row.get('home_pitcher_era', 4.30)
            features['away_pitcher_era']     = row.get('away_pitcher_era', 4.30)
            features['pitcher_era_diff']     = row.get('pitcher_era_diff',
                features['away_pitcher_era'] - features['home_pitcher_era'])
            features['home_run_diff_momentum'] = features.get('home_run_diff_L5',0) - features.get('home_run_diff_L25',0)
            features['away_run_diff_momentum'] = features.get('away_run_diff_L5',0) - features.get('away_run_diff_L25',0)

            all_features.append(features)

        features_df = pd.DataFrame(all_features)
        features_df['home_win']  = df['home_win'].values
        features_df['total_runs'] = df['total_runs'].values
        features_df['run_diff']  = df['run_diff'].values
        features_df['date']      = df['date'].values
        features_df['home_team'] = df['home_team'].values
        features_df['away_team'] = df['away_team'].values
        return features_df

    def _compute_team_rolling(self, prev_games, team, prefix):
        features = {}
        if prev_games.empty:
            for w in self.windows:
                features[f'{prefix}_runs_scored_L{w}']  = 4.5
                features[f'{prefix}_runs_allowed_L{w}'] = 4.5
                features[f'{prefix}_win_pct_L{w}']      = 0.5
                features[f'{prefix}_run_diff_L{w}']     = 0.0
            return features

        runs_scored, runs_allowed, wins = [], [], []
        for _, game in prev_games.iterrows():
            if game['home_team'] == team:
                runs_scored.append(game['home_runs'])
                runs_allowed.append(game['away_runs'])
                wins.append(1 if game['home_runs'] > game['away_runs'] else 0)
            else:
                runs_scored.append(game['away_runs'])
                runs_allowed.append(game['home_runs'])
                wins.append(1 if game['away_runs'] > game['home_runs'] else 0)

        rs = np.array(runs_scored)
        ra = np.array(runs_allowed)
        w  = np.array(wins)

        for window in self.windows:
            n = min(window, len(rs))
            features[f'{prefix}_runs_scored_L{window}']  = rs[-n:].mean() if n > 0 else 4.5
            features[f'{prefix}_runs_allowed_L{window}'] = ra[-n:].mean() if n > 0 else 4.5
            features[f'{prefix}_win_pct_L{window}']      = w[-n:].mean()  if n > 0 else 0.5
            features[f'{prefix}_run_diff_L{window}']     = (rs[-n:]-ra[-n:]).mean() if n > 0 else 0.0
        return features

    def compute_sample_weights(self, dates):
        dates = pd.to_datetime(dates)
        reference = dates.max()
        days_ago = (reference - dates).dt.days
        weights = np.exp(-0.5 * days_ago / self.decay_half_life)
        recent_mask = days_ago <= 14
        weights[recent_mask] *= (1 + self.decay_boost)
        weights = weights / weights.mean()
        return weights.values


# ── 2. POISSON ────────────────────────────────────────────────────────────────

class PoissonModel:
    """Modelo de Poisson para carreras esperadas."""

    def __init__(self):
        self.league_avg = 4.5
        self.home_adv   = 0.15

    def estimate_lambdas(self, features):
        ho = features.get('home_runs_scored_L10', self.league_avg)
        hd = features.get('home_runs_allowed_L10', self.league_avg)
        ao = features.get('away_runs_scored_L10', self.league_avg)
        ad = features.get('away_runs_allowed_L10', self.league_avg)
        hera = features.get('home_starter_era', 4.50)
        aera = features.get('away_starter_era', 4.50)

        lam_h = np.clip(
            (ho/self.league_avg)*(ad/self.league_avg)*(aera/self.league_avg)*self.league_avg*(1+self.home_adv*0.5),
            2.0, 8.0
        )
        lam_a = np.clip(
            (ao/self.league_avg)*(hd/self.league_avg)*(hera/self.league_avg)*self.league_avg*(1-self.home_adv*0.5),
            2.0, 8.0
        )
        return lam_h, lam_a

    def predict_proba(self, features):
        lam_h, lam_a = self.estimate_lambdas(features)
        p_home = p_away = p_tie = 0.0
        for h in range(15):
            for a in range(15):
                p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
                if h > a:   p_home += p
                elif a > h: p_away += p
                else:       p_tie  += p
        p_home += p_tie * 0.55
        p_away += p_tie * 0.45
        return p_home, p_away

    def predict_total(self, features):
        lh, la = self.estimate_lambdas(features)
        return lh + la

    def predict_spread(self, features):
        lh, la = self.estimate_lambdas(features)
        return lh - la

    def calc_over_prob(self, features, line):
        lh, la = self.estimate_lambdas(features)
        return round(1 - poisson.cdf(int(line), lh + la), 4)


# ── 3. ENSEMBLE v3 ────────────────────────────────────────────────────────────

class MLBEnsembleV3:
    """
    Ensemble Poisson + XGBoost + DecisionTree + LogisticRegression.
    Interfaz compatible con inference_pipeline.py.
    """

    def __init__(self, save_dir=None):
        self.fe       = FeatureEngineer()
        self.poisson  = PoissonModel()
        self.scaler   = StandardScaler()
        self.models   = {}
        self.cal      = {}
        self.weights  = {'xgboost':0.35,'decision_tree':0.15,'logistic_regression':0.20,'poisson':0.30}
        self.feat_cols= None
        self.trained  = False
        self.metrics  = {}
        self.save_dir = save_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models'
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, games_df):
        logger.info("Construyendo features v3...")
        fdf = self.fe.build_features(games_df)
        fdf = fdf.iloc[25:].reset_index(drop=True)

        excl = ['home_win','total_runs','run_diff','date','home_team','away_team']
        self.feat_cols = [c for c in fdf.columns if c not in excl and fdf[c].dtype in ['float64','int64']]

        X = fdf[self.feat_cols].fillna(0)
        y = fdf['home_win']
        w = self.fe.compute_sample_weights(fdf['date'])
        Xs = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feat_cols)

        logger.info(f"Features: {len(self.feat_cols)} | Samples: {len(X)} | HomeWin%: {y.mean():.3f}")

        # Time series CV
        tscv = TimeSeriesSplit(n_splits=5)
        fold_accs = []
        for fold, (tr, va) in enumerate(tscv.split(Xs)):
            mods = self._build()
            for n, m in mods.items():
                if n == 'poisson': continue
                try:    m.fit(Xs.iloc[tr], y.iloc[tr], sample_weight=w[tr])
                except: m.fit(Xs.iloc[tr], y.iloc[tr])
            preds = self._predict_ensemble(mods, Xs.iloc[va], fdf.iloc[va])
            acc = accuracy_score(y.iloc[va], (preds > 0.5).astype(int))
            fold_accs.append(acc)
            logger.info(f"  Fold {fold+1}: Acc={acc:.3f}")

        # Final fit
        self.models = self._build()
        for n, m in self.models.items():
            if n == 'poisson': continue
            try:    m.fit(Xs, y, sample_weight=w)
            except: m.fit(Xs, y)

        # Calibración
        for n, m in self.models.items():
            if n == 'poisson':
                self.cal[n] = m; continue
            try:
                c = CalibratedClassifierCV(m, cv=3, method='isotonic')
                c.fit(Xs, y)
                self.cal[n] = c
            except:
                self.cal[n] = m

        preds = self._predict_ensemble(self.cal, Xs, fdf)
        self.metrics = {
            'cv_accuracy': float(np.mean(fold_accs)),
            'train_accuracy': float(accuracy_score(y, (preds>0.5).astype(int))),
            'n_features': len(self.feat_cols),
            'n_samples': len(X),
        }
        self.trained = True
        self._save()
        logger.info(f"✅ Ensemble v3 entrenado — CV Acc: {self.metrics['cv_accuracy']:.3f}")
        return self.metrics

    def _build(self):
        mods = {}
        if XGBOOST:
            mods['xgboost'] = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, min_child_weight=5, gamma=0.1,
                eval_metric='logloss', verbosity=0, random_state=42,
            )
        else:
            mods['xgboost'] = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
        mods['decision_tree'] = DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, min_samples_split=50,
            max_features='sqrt', random_state=42,
        )
        mods['logistic_regression'] = LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42,
        )
        mods['poisson'] = self.poisson
        return mods

    def _predict_ensemble(self, models, X, fdf=None):
        preds = {}
        for name, model in models.items():
            if name == 'poisson':
                if fdf is not None:
                    pp = [self.poisson.predict_proba(fdf.iloc[i].to_dict())[0] for i in range(len(X))]
                    preds[name] = np.array(pp)
                else:
                    preds[name] = np.full(len(X), 0.55)
            else:
                try:
                    preds[name] = model.predict_proba(X)[:,1]
                except: continue

        out = np.zeros(len(X)); tw = 0
        for n, p in preds.items():
            w = self.weights.get(n, 0.25)
            out += w * p; tw += w
        return out / tw

    # ── Predict (interface pública) ───────────────────────────────────────────

    def predict_with_confidence(self, df: pd.DataFrame, threshold=0.57):
        """
        Compatible con inference_pipeline.py.
        Retorna (probs_array, confident_mask_array).
        """
        if not self.trained:
            # Fallback solo Poisson
            probs = np.array([
                self.poisson.predict_proba(row.to_dict())[0]
                for _, row in df.iterrows()
            ])
            confident = (probs > threshold) | (probs < (1 - threshold))
            return probs, confident

        X = df.reindex(columns=self.feat_cols, fill_value=0).fillna(0)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=self.feat_cols)
        probs = self._predict_ensemble(self.cal, Xs, df)

        # Confident = todos los modelos coinciden + prob fuera de umbral
        confident = np.zeros(len(probs), dtype=bool)
        for i, (prob, (_, row)) in enumerate(zip(probs, df.iterrows())):
            indiv = {}
            for name, model in self.cal.items():
                if name == 'poisson':
                    indiv[name] = self.poisson.predict_proba(row.to_dict())[0]
                else:
                    try:
                        indiv[name] = model.predict_proba(
                            pd.DataFrame([row.reindex(self.feat_cols, fill_value=0).fillna(0)])
                        )[0][1]
                    except: pass
            if indiv:
                vals = list(indiv.values())
                all_same = all(v > 0.5 for v in vals) or all(v < 0.5 for v in vals)
                confident[i] = all_same and np.std(vals) < 0.08 and abs(prob - 0.5) > (threshold - 0.5)

        return probs, confident

    def calculate_edge(self, model_prob, market_odds):
        if market_odds > 0:
            implied = 100 / (market_odds + 100)
            decimal = (market_odds / 100) + 1
        else:
            implied = abs(market_odds) / (abs(market_odds) + 100)
            decimal = (100 / abs(market_odds)) + 1

        edge    = model_prob - implied
        ev      = (model_prob * decimal) - 1
        kelly_q = max(0, ((model_prob * decimal - 1) / (decimal - 1)) * 0.25)

        if edge * 100 > 8:   grade = '🔥 STRONG BET'
        elif edge * 100 > 5: grade = '✅ GOOD VALUE'
        elif edge * 100 > 3: grade = '⚠️ MARGINAL'
        else:                grade = '❌ NO VALUE'

        return {'edge_pct': round(edge*100,2), 'ev': round(ev,4),
                'kelly_quarter': round(kelly_q,4), 'implied_prob': round(implied,4),
                'model_prob': round(model_prob,4), 'decimal_odds': round(decimal,3),
                'grade': grade, 'bet_size_pct': round(kelly_q*100,2)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, 'ensemble_v3.joblib')
        joblib.dump({
            'models': self.models, 'cal': self.cal, 'scaler': self.scaler,
            'feat_cols': self.feat_cols, 'weights': self.weights,
            'metrics': self.metrics, 'trained': True,
        }, path)
        logger.info(f"💾 Guardado: {path}")

    def load(self):
        path = os.path.join(self.save_dir, 'ensemble_v3.joblib')
        if os.path.exists(path):
            d = joblib.load(path)
            self.models = d['models']; self.cal = d['cal']
            self.scaler = d['scaler']; self.feat_cols = d['feat_cols']
            self.weights = d['weights']; self.metrics = d['metrics']
            self.trained = True
            logger.info("✅ ensemble_v3.joblib cargado")
            return True
        return False
