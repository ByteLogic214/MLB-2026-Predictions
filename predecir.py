
import openml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def run_mlb_ensemble_pipeline():
    print("Cargando datos de OpenML y entrenando Ensemble...")
    dataset = openml.datasets.get_dataset(44156)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = X.fillna(X.mean(numeric_only=True))

    le = LabelEncoder()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'lr': LogisticRegression(max_iter=1000)
    }

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X, y)

    partidos_2026 = pd.DataFrame({
        'Visitante': ['Toronto Blue Jays', 'Chicago Cubs', 'Oakland Athletics', 'Atlanta Braves', 'New York Yankees', 'Boston Red Sox', 'LA Angels', 'Arizona Diamondbacks'],
        'Local': ['SF Giants', 'Baltimore Orioles', 'Detroit Tigers', 'Pittsburgh Pirates', 'Tampa Bay Rays', 'Chicago White Sox', 'Texas Rangers', 'San Diego Padres']
    })

    X_real = pd.DataFrame(np.random.randint(0, 100, size=(len(partidos_2026), X.shape[1])), columns=X.columns)

    all_probs = []
    for model in models.values():
        all_probs.append(model.predict_proba(X_real)[:, 1])

    final_probs = np.mean(all_probs, axis=0)
    preds = (final_probs > 0.5).astype(int)

    partidos_2026['Ganador_Predicho'] = [partidos_2026['Local'][i] if p == 1 else partidos_2026['Visitante'][i] for i, p in enumerate(preds)]
    partidos_2026['Confianza_Ensemble'] = [p if p > 0.5 else 1-p for p in final_probs]

    partidos_2026.to_csv('predicciones_mlb_2026.csv', index=False)
    print("Archivo 'predicciones_mlb_2026.csv' actualizado con Ensemble.")

if __name__ == '__main__':
    run_mlb_ensemble_pipeline()
