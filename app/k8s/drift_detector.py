import requests
import time

PROMETHEUS_URL = "http://localhost:9090"
CONFIDENCE_THRESHOLD = 0.65
CHECK_INTERVAL_SECONDS = 30

def get_avg_confidence():
    query = "sum(increase(prediction_confidence_sum[5m])) / sum(increase(prediction_confidence_count[5m]))"
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
    result = resp.json()["data"]["result"]
    if not result:
        return None
    return float(result[0]["value"][1])

def trigger_retrain():
    print("DRIFT DETECTED - retrain pipeline would trigger here (Stage 5 will wire this up for real)")

if __name__ == "__main__":
    print("Drift detector started. Watching model confidence...")
    while True:
        avg_conf = get_avg_confidence()
        if avg_conf is None:
            print("No confidence data yet - waiting for more predictions...")
        else:
            print(f"Current average confidence: {avg_conf:.3f}")
            if avg_conf < CONFIDENCE_THRESHOLD:
                trigger_retrain()
        time.sleep(CHECK_INTERVAL_SECONDS)