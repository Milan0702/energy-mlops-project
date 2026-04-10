# MLOps Concepts Explained

This document explains each concept used in the Smart Energy Consumption Prediction System and shows how it's implemented in the code. Perfect for team members who want to understand the underlying principles.

---

## Table of Contents

1. [Data Preprocessing](#1-data-preprocessing)
2. [Feature Engineering](#2-feature-engineering)
3. [Machine Learning Pipeline](#3-machine-learning-pipeline)
4. [Experiment Tracking with MLflow](#4-experiment-tracking-with-mlflow)
5. [REST API with FastAPI](#5-rest-api-with-fastapi)
6. [Database Integration with PostgreSQL](#6-database-integration-with-postgresql)
7. [Prometheus Metrics](#7-prometheus-metrics)
8. [Grafana Dashboards](#8-grafana-dashboards)
9. [Docker Containerization](#9-docker-containerization)
10. [Docker Compose Orchestration](#10-docker-compose-orchestration)
11. [Reverse Proxy with Nginx](#11-reverse-proxy-with-nginx)
12. [CI/CD with Jenkins](#12-cicd-with-jenkins)

---

## 1. Data Preprocessing

### Concept
Data preprocessing is the process of cleaning and transforming raw data into a format suitable for machine learning. This includes handling missing values, converting data types, and filtering invalid records.

### Why It's Important
- Raw data often contains missing values, inconsistencies, or incorrect formats
- ML models require consistent, numeric input
- Preprocessing improves model accuracy

### Implementation in This Project

**File:** `src/preprocess.py`

```python
import pandas as pd

# 1. Load data with semicolon separator
df = pd.read_csv(RAW_PATH, sep=";", low_memory=False)

# 2. Handle missing values ("?" string)
df.replace("?", pd.NA, inplace=True)

# 3. Convert to numeric (coerce errors to NaN)
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

# 4. Drop rows with missing target
df = df.dropna(subset=["Global_active_power"])
```

**Key techniques used:**
- `sep=";"` - UCI dataset uses semicolon separator
- `low_memory=False` - Prevents mixed type inference
- `pd.NA` - Modern pandas missing value representation
- `dropna()` - Remove incomplete records

---

## 2. Feature Engineering

### Concept
Feature engineering is the process of creating new features from raw data that better represent the underlying problem to the ML model. For time-series prediction, extracting temporal features is crucial.

### Why It's Important
- Raw datetime values aren't useful for ML directly
- Temporal patterns (hour of day, weekend) often predict consumption
- Good features = better model performance

### Implementation in This Project

**File:** `src/preprocess.py`

```python
# Combine Date + Time into single datetime
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], 
                                  format="%d/%m/%Y %H:%M:%S")

# Extract time-based features
df["hour_of_day"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek

# Create binary features
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_peak_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
```

**Features created:**
| Feature | Type | Description |
|---------|------|-------------|
| `hour_of_day` | Integer (0-23) | Hour of the day |
| `day_of_week` | Integer (0-6) | Day of week (Monday=0) |
| `is_weekend` | Binary (0/1) | 1 if Saturday/Sunday |
| `is_peak_hour` | Binary (0/1) | 1 if 7-9am or 5-8pm |

---

## 3. Machine Learning Pipeline

### Concept
A pipeline chains multiple transformation steps together, ensuring consistent processing during both training and inference. It's a best practice in ML engineering.

### Why It's Important
- Ensures identical preprocessing at train and predict time
- Prevents data leakage (e.g., fitting scaler on test data)
- Makes deployment simpler

### Implementation in This Project

**File:** `src/train.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Create pipeline with two steps
pipeline = Pipeline([
    ("scaler", StandardScaler()),      # Step 1: Normalize features
    ("model", LinearRegression())      # Step 2: Train model
])

# Fit on training data (both steps applied automatically)
pipeline.fit(X_train, y_train)

# Predict (both steps applied automatically)
prediction = pipeline.predict(X_test)
```

**Pipeline flow:**
```
Input (4 features) → StandardScaler → LinearRegression → Output (1 prediction)
```

**Why LinearRegression?**
- Simple and interpretable
- Fast to train (no GPU needed)
- Good baseline for time-series data

---

## 4. Experiment Tracking with MLflow

### Concept
Experiment tracking records all details of ML experiments: parameters, metrics, artifacts, and environment. This enables reproducibility and comparison between runs.

### Why It's Important
- Track what worked and what didn't
- Reproduce any previous model
- Compare metrics across experiments

### Implementation in This Project

**File:** `src/train.py`

```python
import mlflow

# Set tracking server
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("energy-prediction")

# Start run and log everything
with mlflow.start_run():
    # Log parameters (hyperparameters)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("features", "hour_of_day,day_of_week,is_weekend,is_peak_hour")
    mlflow.log_param("train_size", 0.8)
    
    # Log metrics (evaluation results)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
```

**What's logged:**
- **Parameters**: model_type, features, train_size
- **Metrics**: RMSE, MAE, R²
- **Artifacts**: model.pkl (saved separately)

**MLflow UI:** Access at http://localhost:5000

---

## 5. REST API with FastAPI

### Concept
A REST API (Representational State Transfer) allows clients to interact with the model over HTTP. FastAPI is a modern Python web framework that automatically generates documentation.

### Why It's Important
- Enables web and mobile applications to use the model
- Provides standardization (HTTP methods, JSON)
- FastAPI offers automatic documentation

### Implementation in This Project

**File:** `api/main.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Energy Prediction API")

# Define request model
class PredictRequest(BaseModel):
    hour_of_day: int
    day_of_week: int
    is_weekend: int | None = None
    is_peak_hour: int | None = None

# Define endpoint
@app.post("/predict-energy")
async def predict_energy(req: PredictRequest):
    # Auto-compute optional fields
    if req.is_weekend is None:
        req.is_weekend = 1 if req.day_of_week >= 5 else 0
    
    # Run inference
    features = [[req.hour_of_day, req.day_of_week, req.is_weekend, req.is_peak_hour]]
    prediction = model.predict(features)
    
    return {"predicted_energy_consumption": prediction[0], "unit": "kW"}
```

**Key FastAPI features:**
- `@app.post()` - Decorator to create POST endpoint
- `BaseModel` - Pydantic model for request validation
- Automatic docs at `/docs`

---

## 6. Database Integration with PostgreSQL

### Concept
A database stores structured data persistently. PostgreSQL is a powerful open-source relational database. Every prediction request is logged for audit and analysis.

### Why It's Important
- Store prediction history
- Enable analysis of prediction patterns
- Provide data for dashboards

### Implementation in This Project

**File:** `api/main.py`

```python
from sqlalchemy import create_engine, Table, Column, Integer, Float, DateTime

# Create database connection
DATABASE_URL = "postgresql://user:password@postgres:5432/energydb"
engine = create_engine(DATABASE_URL)

# Define table schema
predictions_table = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", DateTime, default=datetime.now),
    Column("hour_of_day", Integer),
    Column("day_of_week", Integer),
    Column("is_weekend", Integer),
    Column("is_peak_hour", Integer),
    Column("prediction", Float)
)

# Insert prediction
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
    conn.commit()  # Important: commit to save!
```

**Database schema:**
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend INTEGER,
    is_peak_hour INTEGER,
    prediction FLOAT
);
```

---

## 7. Prometheus Metrics

### Concept
Prometheus is a metrics collection system that periodically scrapes (pulls) data from applications. It stores time-series data that can be queried and visualized.

### Why It's Important
- Monitor application health and performance
- Track request rates, latency, errors
- Enable alerting on anomalies

### Implementation in This Project

**File:** `api/main.py`

```python
from prometheus_fastapi_instrumentator import Instrumentator

# Create FastAPI app
app = FastAPI()

# Add Prometheus instrumentation (3 lines!)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

**What gets automatically tracked:**
- `http_requests_total` - Total request count
- `http_request_duration_seconds` - Request duration
- `http_response_size_bytes` - Response size

**Prometheus config** (`monitoring/prometheus.yml`):
```yaml
scrape_configs:
  - job_name: "energy-api"
    static_configs:
      - targets: ["api:8000"]
```

---

## 8. Grafana Dashboards

### Concept
Grafana is a visualization tool that connects to data sources (like Prometheus) and creates interactive dashboards with charts, graphs, and alerts.

### Why It's Important
- Visualize system performance
- Monitor trends over time
- Create alerts for anomalies

### Implementation in This Project

**Dashboard panels:**
1. **Request Rate**: `rate(http_requests_total[1m])`
2. **P95 Latency**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
3. **Total Predictions**: `increase(http_requests_total{handler="/predict-energy"}[24h])`

**Access:** http://localhost:3000 (admin/admin)

---

## 9. Docker Containerization

### Concept
Docker packages an application with all its dependencies into a container - a lightweight, standalone executable. This ensures the app runs the same everywhere.

### Why It's Important
- Consistent environment (dev = prod)
- Isolation between services
- Easy scaling

### Implementation in This Project

**Dockerfile.api:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt
COPY api/ ./api/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile.trainer:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
CMD ["python", "src/train.py"]
```

**Key concepts:**
- `FROM` - Base image
- `COPY` - Add files to container
- `RUN` - Execute commands during build
- `EXPOSE` - Document ports
- `CMD` - Command to run

---

## 10. Docker Compose Orchestration

### Concept
Docker Compose manages multi-container applications. A single YAML file defines all services, their relationships, and how they connect.

### Why It's Important
- Define entire stack in one file
- Manage dependencies between services
- Shared volumes for data exchange

### Implementation in This Project

**File:** `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models

volumes:
  postgres_data:
```

**Key features:**
- `depends_on` - Start order control
- `condition: service_healthy` - Wait for healthcheck
- `volumes` - Share data between host and container

---

## 11. Reverse Proxy with Nginx

### Concept
A reverse proxy sits between clients and servers, forwarding requests to backend services. It provides a single entry point and can handle load balancing, SSL, and caching.

### Why It's Important
- Single URL for multiple services
- Hide backend architecture
- Handle static files efficiently

### Implementation in This Project

**File:** `nginx/nginx.conf`

```nginx
http {
    upstream fastapi {
        server api:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://fastapi;
        }
    }
}
```

**How it works:**
```
Browser → localhost:80 → Nginx → api:8000 → FastAPI
```

---

## 12. CI/CD with Jenkins

### Concept
CI/CD (Continuous Integration/Continuous Deployment) automates the build, test, and deployment process. Jenkins is a popular automation server that runs pipelines defined in code.

### Why It's Important
- Automated testing and deployment
- Consistent releases
- Faster development cycles

### Implementation in This Project

**File:** `Jenkinsfile`

```groovy
pipeline {
    agent any
    
    stages {
        stage("Checkout") {
            steps {
                checkout scm
            }
        }
        
        stage("Build Docker Images") {
            steps {
                bat "docker compose -p ${PROJECT_NAME} build"
            }
        }
        
        stage("Run Training") {
            steps {
                bat "docker compose -p ${PROJECT_NAME} run --rm trainer"
            }
        }
        
        stage("Deploy All Services") {
            steps {
                bat "docker compose -p ${PROJECT_NAME} up -d"
            }
        }
    }
}
```

**Pipeline stages:**
1. **Checkout** - Pull code from GitHub
2. **Start MLflow** - Ensure tracking server is running
3. **Build Docker Images** - Create containers
4. **Run Training** - Train model
5. **Deploy All Services** - Start the stack

---

## Summary: How Everything Connects

```
┌─────────────────────────────────────────────────────────────────┐
│                     JENKINS PIPELINE                            │
│  1. Checkout code from GitHub                                   │
│  2. Build Docker images                                         │
│  3. Run training → MLflow logs metrics                          │
│  4. Deploy all services via Docker Compose                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DOCKER COMPOSE                              │
│  - postgres (database)                                          │
│  - mlflow (experiment tracking)                                 │
│  - api (FastAPI)                                                 │
│  - nginx (reverse proxy)                                         │
│  - prometheus (metrics)                                          │
│  - grafana (dashboards)                                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     USER FLOW                                   │
│  1. User opens http://localhost                                 │
│  2. Nginx proxies to FastAPI                                    │
│  3. FastAPI serves HTML dashboard                               │
│  4. User submits prediction                                     │
│  5. FastAPI runs model inference                                │
│  6. Prediction logged to PostgreSQL                             │
│  7. Prometheus scrapes metrics                                  │
│  8. Grafana visualizes in dashboard                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Further Reading

- **scikit-learn Pipelines**: https://scikit-learn.org/stable/modules/compose.html
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://docs.docker.com/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **Jenkins**: https://www.jenkins.io/doc/

---

*This document explains the core MLOps concepts demonstrated in the Energy Prediction project. Each section shows the theory, the why, and the actual implementation code.*
