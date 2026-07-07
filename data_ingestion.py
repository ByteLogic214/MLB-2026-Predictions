import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

class DataIngestor:
    def __init__(self):
        self.base_path = 'MLB-2026-Predictions/'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        self.sources = {
            'baseball_ref': 'https://www.baseball-reference.com/leagues/majors/2024.shtml',
            'espn_teams': 'https://www.espn.com/mlb/stats/team/_/season/2024/seasontype/2',
            'fangraphs': 'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2024'
        }

    def fetch_espn_stats(self):
        print('Obteniendo estadísticas de ESPN...')
        try:
            response = requests.get(self.sources['espn_teams'], headers=self.headers)
            tables = pd.read_html(response.text)
            df = tables[0]
            df.to_csv(f'{self.base_path}espn_raw.csv', index=False)
            return df
        except Exception as e:
            print(f'Error en ESPN: {e}')
            return pd.DataFrame()

    def fetch_baseball_reference(self):
        print('Obteniendo datos de Baseball-Reference...')
        try:
            response = requests.get(self.sources['baseball_ref'], headers=self.headers)
            tables = pd.read_html(response.text)
            # Generalmente la tabla 0 es el resumen de la liga
            df = tables[0]
            df.to_csv(f'{self.base_path}bref_raw.csv', index=False)
            return df
        except Exception as e:
            print(f'Error en B-Ref: {e}')
            return pd.DataFrame()

    def clean_and_process(self, df_list):
        print('Limpiando y unificando datos...')
        # Unificación básica por equipo para el modelo Random Forest
        if not df_list: return
        consolidated = pd.concat(df_list, axis=0, ignore_index=True).fillna(0)
        consolidated.to_csv(f'{self.base_path}datos_entrenamiento_mlb.csv', index=False)
        return consolidated

    def run_daily_update(self):
        print('--- INICIANDO ACTUALIZACIÓN MULTI-FUENTE ---')
        data = []
        data.append(self.fetch_espn_stats())
        data.append(self.fetch_baseball_reference())
        
        valid_data = [d for d in data if not d.empty]
        if valid_data:
            self.clean_and_process(valid_data)
            print('✅ Base de datos actualizada con éxito.')
            return True
        return False

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.run_daily_update()
