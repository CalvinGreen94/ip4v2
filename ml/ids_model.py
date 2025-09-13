from sklearn.ensemble import IsolationForest
import joblib

class IntrusionDetector:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.1)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)  # Lower = more anomalous
        print(f"Predictions: {predictions}")
        print(f"Anomaly scores: {scores}")
        return predictions, scores
    
    def get_anomalies(self, X):
        preds = self.model.predict(X)
        return X[preds == -1]
    

    def save_model(self, path='ids_model.pkl'):
        joblib.dump(self.model, path)

    def load_model(self, path='ids_model.pkl'):
        self.model = joblib.load(path)
