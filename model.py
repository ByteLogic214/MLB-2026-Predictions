from sklearn.ensemble import RandomForestClassifier

class ModeloPredictivo:
    def __init__(self):
        self.modelo = RandomForestClassifier()
    def entrenar(self, X, y): self.modelo.fit(X, y)
    def predecir(self, X): return self.modelo.predict(X)
