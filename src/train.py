# what it does: Load processed data, train LinearRegression model, log metrics to MLflow, save model
# input: data/processed/features.csv
# output: models/model.pkl

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import joblib

DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/model.pkl"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "energy-prediction"

df = pd.read_csv(DATA_PATH)

X = df[["hour_of_day", "day_of_week", "is_weekend", "is_peak_hour"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("features", "hour_of_day,day_of_week,is_weekend,is_peak_hour")
    mlflow.log_param("train_size", 0.8)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
