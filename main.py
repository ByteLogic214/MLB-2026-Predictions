import logging
from config import Config
from api import OddsAPI
from model import ModeloPredictivo

def run():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('🚀 MLB PREDICTOR v2.0 - LIVE')
    
    config = Config()
    config.validate()
    
    modelo = ModeloPredictivo()
    modelo.reentrenar()
    
    api = OddsAPI(config)
    cuotas = api.obtener_cuotas()
    
    print(f'Procesadas {len(cuotas)} oportunidades de mercado.')

if __name__ == '__main__':
    run()
