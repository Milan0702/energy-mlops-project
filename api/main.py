import os
import json
import logging
from datetime import datetime

import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, Float, DateTime, String
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/energydb")

app = FastAPI(title="Energy Prediction API")

Instrumentator().instrument(app).expose(app)

engine = create_engine(DATABASE_URL)
metadata = MetaData()

predictions_table = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, default=datetime.now),
    Column("hour_of_day", Integer),
    Column("day_of_week", Integer),
    Column("is_weekend", Integer),
    Column("is_peak_hour", Integer),
    Column("prediction", Float)
)

def insert_prediction(conn, hour, day, is_weekend, is_peak_hour, prediction_value):
    conn.execute(
        predictions_table.insert().values(
            hour_of_day=hour,
            day_of_week=day,
            is_weekend=is_weekend,
            is_peak_hour=is_peak_hour,
            prediction=prediction_value
        )
    )
    conn.commit()

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        logger.warning("model.pkl not found — API will return 503 until model is ready")
        return None

def load_metrics():
    try:
        with open("models/metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@app.on_event("startup")
async def startup():
    app.state.model = load_model()
    app.state.metrics = load_metrics()
    metadata.create_all(engine)
    logger.info("API startup complete")

class PredictRequest(BaseModel):
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int | None = None
    is_peak_hour: int | None = None

class PredictResponse(BaseModel):
    predicted_energy_consumption: float
    unit: str = "kW"
    timestamp: str

@app.get("/")
async def root():
    with open("api/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict-energy", response_model=PredictResponse)
async def predict_energy(req: PredictRequest):
    if req.is_weekend is None:
        req.is_weekend = 1 if req.day_of_week >= 5 else 0
    if req.is_peak_hour is None:
        req.is_peak_hour = 1 if req.hour_of_day in [7, 8, 9, 17, 18, 19, 20] else 0

    features = np.array([[req.hour_of_day, req.day_of_week, req.is_weekend, req.is_peak_hour]])

    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prediction = float(app.state.model.predict(features)[0])

    with engine.connect() as conn:
        insert_prediction(conn, req.hour_of_day, req.day_of_week, req.is_weekend, req.is_peak_hour, prediction)

    return PredictResponse(
        predicted_energy_consumption=prediction,
        timestamp=datetime.now().isoformat()
    )

@app.get("/predictions")
async def get_predictions():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10"))
        rows = result.fetchall()
    predictions = []
    for row in rows:
        predictions.append({
            "id": row[0],
            "timestamp": str(row[1]),
            "hour_of_day": row[2],
            "day_of_week": row[3],
            "is_weekend": row[4],
            "is_peak_hour": row[5],
            "prediction": row[6]
        })
    return {"predictions": predictions}

@app.get("/stats")
async def get_stats():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
        count = result.scalar()
    
    metrics = getattr(app.state, "metrics", None)
    model_info = {
        "model_loaded": app.state.model is not None,
        "accuracy": metrics.get("accuracy_percent") if metrics else None,
        "r2": metrics.get("r2") if metrics else None,
        "rmse": metrics.get("rmse") if metrics else None,
        "mae": metrics.get("mae") if metrics else None
    }
    return {"total_predictions": count, "model": model_info}

@app.get("/model-metrics")
async def get_model_metrics():
    metrics = getattr(app.state, "metrics", None)
    if metrics is None:
        return {"error": "Metrics not available"}
    return metrics

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}

@app.post("/reload-model")
async def reload_model():
    try:
        app.state.model = load_model()
        app.state.metrics = load_metrics()
        return {"status": "reloaded", "metrics": app.state.metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
