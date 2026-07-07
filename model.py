import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class MLBQuantModel:
    def __init__(self):
        # Modelo para ganador
        self.clf = RandomForestClassifier(n_estimators=100)
        # Modelo para total de carreras (Regresión)
        self.reg = RandomForestRegressor(n_estimators=100)

    def entrenar_con_contexto(self, data_path):
        if not os.path.exists(data_path): return
        df = pd.read_csv(data_path)
        
        # Lógica para Ganador
        X = df[['team_era', 'opp_era', 'venue_factor', 'avg_runs_last_10']]
        y_winner = df['winner_label']
        self.clf.fit(X, y_winner)
        
        # Lógica para Totales
        y_total = df['total_runs']
        self.reg.fit(X, y_total)

    def predecir_todo(self, features):
        prob_win = self.clf.predict_proba(features)[0][1]
        pred_runs = self.reg.predict(features)[0]
        return prob_win, pred_runs
