import joblib
import numpy as np

class DriftAwareModel:
    def __init__(self, model_path="model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, features: np.ndarray):
        proba = self.model.predict_proba(features)
        confidence = float(np.max(proba))
        prediction = int(np.argmax(proba))
        return prediction, confidence