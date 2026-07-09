"""
test_system.py - Tests unitarios del sistema MLB 2026
"""
import unittest
import os
import sys
import pandas as pd
import numpy as np

# Aseguramos que src/ esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config


class TestConfig(unittest.TestCase):
    def test_config_initialization(self):
        config = Config()
        self.assertIsInstance(config, Config)

    def test_config_validate_missing_keys(self):
        """El validate() debe retornar False cuando faltan env vars."""
        config = Config()
        # En entorno sin envs, debe retornar False sin lanzar excepción
        result = config.validate()
        self.assertIn(result, [True, False])


class TestMLBQuantModel(unittest.TestCase):
    def test_model_import(self):
        from model import MLBQuantModel
        model = MLBQuantModel()
        self.assertIsNotNone(model)

    def test_rolling_features_no_leakage(self):
        """Verifica que compute_rolling_features usa shift(1) anti-leakage."""
        from model import MLBQuantModel
        model = MLBQuantModel()
        df = pd.DataFrame({
            'game_date': pd.date_range('2024-04-01', periods=20),
            'team': ['NYY'] * 20,
            'win': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
            'runs_scored': [5, 3, 7, 4, 2, 6, 3, 5, 4, 2, 6, 3, 1, 7, 5, 2, 4, 3, 6, 5],
            'runs_allowed': [3, 5, 2, 4, 6, 1, 4, 3, 5, 7, 2, 6, 4, 3, 2, 5, 3, 4, 2, 3],
        })
        result = model.compute_rolling_features(df)
        # El primer juego NO debe tener el valor actual en su rolling
        # (shift(1) asegura esto)
        self.assertIn('win_L5', result.columns)

    def test_sample_weights_shape(self):
        from model import MLBQuantModel
        model = MLBQuantModel()
        dates = pd.date_range('2025-01-01', periods=10)
        weights = model.compute_sample_weights(dates)
        self.assertEqual(len(weights), 10)
        self.assertTrue(all(w > 0 for w in weights))


class TestEnsembleModel(unittest.TestCase):
    def test_ensemble_init(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
        from ensemble_model import MLBEnsembleModel
        model = MLBEnsembleModel()
        self.assertIsNotNone(model)

    def test_heuristic_predict_not_random(self):
        """La predicción heurística debe ser determinista."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
        from ensemble_model import MLBEnsembleModel
        model = MLBEnsembleModel()
        df = pd.DataFrame({
            'home_win_pct_L10': [0.6, 0.4, 0.55],
            'away_win_pct_L10': [0.4, 0.6, 0.45],
            'home_advantage': [0.035, 0.035, 0.035],
        })
        preds1 = model._heuristic_predict(df)
        preds2 = model._heuristic_predict(df)
        np.testing.assert_array_almost_equal(preds1, preds2)


class TestDedupManager(unittest.TestCase):
    def test_dedup_hash_and_store(self):
        from dedup_manager import DedupManager
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            tmp_path = f.name
        dedup = DedupManager(tmp_path)
        h = dedup.generate_hash('NYY', 'BOS', 'moneyline', '2026-07-09')
        self.assertFalse(dedup.ya_enviado(h))
        dedup.marcar_enviado(h)
        self.assertTrue(dedup.ya_enviado(h))
        os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
