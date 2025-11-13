from __future__ import annotations
import json, time, uuid
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path("artifacts")
LOGS_DIR = Path("logs"); LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "inference.jsonl"

# ---- Pydantic schemas ----
class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="[[sepal_len, sepal_wid, petal_len, petal_wid], ...]")

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    model_version: str

# ---- Load model + training stats ----
_model = joblib.load(ARTIFACTS / "model.pkl")
_stats = json.loads((ARTIFACTS / "training_stats.json").read_text())

# ---- Rolling stats for simple drift proxy ----
class Rolling:
    def __init__(self, n_features: int, window: int = 500):
        from collections import deque
        self.buf = deque(maxlen=window)
        self.n = n_features
    def update(self, batch: List[List[float]]):
        for r in batch: self.buf.append(r)
    def mean(self):
        if not self.buf: return [0.0]*self.n
        cols = list(zip(*self.buf))
        return [sum(c)/len(c) for c in cols]

def drift_proxy(baseline_means, current_means) -> float:
    # average relative mean shift; simple & fast to compute
    eps = 1e-6
    diffs = [abs(c - b) / (abs(b) + eps) for b, c in zip(baseline_means, current_means)]
    return float(sum(diffs) / len(diffs))

_roll = Rolling(n_features=4)

app = FastAPI(title="Iris Logistic Regression API", version="0.1.0")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "trained_at": _stats.get("trained_at", "unknown"),
        "metrics": _stats.get("metrics", {}),
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.instances:
        raise HTTPException(status_code=400, detail="Empty instances")
    start = time.time()
    rid = str(uuid.uuid4())

    try:
        preds = _model.predict(req.instances)
        prob = _model.predict_proba(req.instances).tolist()
        _roll.update(req.instances)

        # monitoring
        means = _roll.mean()
        drift = drift_proxy(_stats.get("feature_means", [0,0,0,0]), means)
        rec = {
            "ts": time.time(),
            "request_id": rid,
            "path": "/predict",
            "latency_ms": int((time.time() - start) * 1000),
            "n": len(req.instances),
            "rolling_means": means,
            "drift_proxy": drift,
            "pred_counts": {int(k): int(list(preds).count(k)) for k in set(preds)},
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        return PredictResponse(
            predictions=[int(p) for p in preds],
            probabilities=prob,
            model_version=_stats.get("trained_at", "unknown"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
