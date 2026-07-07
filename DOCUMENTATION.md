# ⚾ MLB 2026 Predictions - Investment Grade System

## Arquitectura Modular (SOLID)
- **config.py**: Gestión centralizada de secretos y constantes.
- **api.py**: Cliente robusto para Odds-API con manejo de timeouts.
- **model.py**: Motor de ML basado en Random Forest con auto-reentrenamiento.
- **data_ingestion.py**: Motor de datos multi-fuente con modo de resiliencia.
- **main.py**: Orquestador principal del flujo de predicción y alertas.

## Seguridad y Estabilidad
- Implementación de **Timeouts** en todas las peticiones externas.
- Logs detallados para depuración en producción.
- Validación de consistencia lógica mediante **Juez Groq (Llama 3.3)**.

## Ejecución Automatizada
El sistema está diseñado para correr vía GitHub Actions de forma diaria, sincronizando el modelo con las cuotas más recientes del mercado.
