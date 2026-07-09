import logging
from src.live_data_fetcher import LiveDataFetcher
from src.models.ensemble_model import MLBEnsembleModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InferencePipeline")

def run_prediction_flow(date_str="2026-07-08"):
    logger.info("--- Iniciando Pipeline de Predicción MLB 2026 ---")
    
    # 1. Ingesta
    fetcher = LiveDataFetcher()
    df_live = fetcher.get_real_mlb_data(date_str)
    
    if df_live is None or df_live.empty:
        logger.error("No se obtuvieron datos para procesar.")
        return

    # 2. Modelo (Ensamble)
    model = MLBEnsembleModel()
    predictions = model.predict(df_live)
    
    df_live['win_probability'] = predictions
    
    # 3. Guardar resultados
    output_path = "data/processed/predictions_final.csv"
    df_live.to_csv(output_path, index=False)
    logger.info(f"✅ Proceso completado. Resultados en: {output_path}")

if __name__ == "__main__":
    run_prediction_flow()