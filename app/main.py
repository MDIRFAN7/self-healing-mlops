from fastapi import FastAPI
from pydantic import BaseModel
from model import DriftAwareModel
import numpy as np
import time
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

app = FastAPI()
model = DriftAwareModel()

PREDICTION_COUNTER = Counter("predictions_total", "Total predictions made")
CONFIDENCE_HISTOGRAM = Histogram("prediction_confidence", "Confidence of predictions")
LATENCY_HISTOGRAM = Histogram("prediction_latency_seconds", "Latency of predictions")

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    features = np.array(req.features).reshape(1, -1)
    prediction, confidence = model.predict(features)
    latency = time.time() - start

    PREDICTION_COUNTER.inc()
    CONFIDENCE_HISTOGRAM.observe(confidence)
    LATENCY_HISTOGRAM.observe(latency)

    return {"prediction": prediction, "confidence": confidence, "latency": latency}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok"}