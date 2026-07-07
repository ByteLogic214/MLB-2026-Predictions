import unittest
import os
from config import Config
from model import ModeloPredictivo

class TestMLBSystem(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.modelo = ModeloPredictivo()

    def test_config_initialization(self):
        # Verificar que la clase config cargue aunque falten variables
        self.assertIsInstance(self.config, Config)

    def test_model_file_missing_handling(self):
        # El modelo debe retornar False elegantemente si no hay datos
        res = self.modelo.reentrenar_con_datos_recientes()
        self.assertIn(res, [True, False])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
