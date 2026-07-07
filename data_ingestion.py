import pandas as pd
import os
import shutil

class DataIngestor:
    def __init__(self):
        self.base_path = 'MLB-2026-Predictions/'
        self.local_master = f'{self.base_path}datos_entrenamiento_mlb.csv'

    def run_daily_update(self):
        print('--- INICIANDO MOTOR DE DATOS (MODO RESILIENCIA) ---')
        if os.path.exists(self.local_master):
            print(f'✅ Utilizando base de datos maestra: {self.local_master}')
            # Simulamos el procesamiento de datos frescos
            df = pd.read_csv(self.local_master)
            df.to_csv(self.local_master, index=False)
            print(f'📊 Motor listo con {len(df)} registros para el entrenamiento.')
            return True
        else:
            print('⚠️ No se encontró la base maestra. Creando estructura inicial...')
            return False

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.run_daily_update()
