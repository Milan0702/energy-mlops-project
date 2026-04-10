# Smart Energy Consumption Prediction System

A complete end-to-end MLOps demonstration project that builds, trains, and serves a machine learning model using modern DevOps practices. This project showcases how to create a production-ready ML pipeline with experiment tracking, API serving, monitoring, and CI/CD automation.

## Overview

This project predicts household electric power consumption based on time-based features like hour of day, day of week, weekend/peak indicators. It demonstrates:

- **Data Preprocessing**: Raw UCI power consumption data → Feature engineering
- **Model Training**: scikit-learn LinearRegression with MLflow experiment tracking
- **API Serving**: FastAPI REST API with prediction logging to PostgreSQL
- **Web Dashboard**: Interactive HTML/CSS/JS UI served by FastAPI
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **CI/CD**: Jenkins pipeline with automated build, train, and deploy

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Browser)                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        NGINX (Port 80)                                   │
│                    Reverse Proxy & Load Balancer                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐    ┌─────────────────────────────────────┐
│    FASTAPI (Port 8000)      │    │   MLflow (Port 5000)                │
│  ┌───────────────────────┐  │    │  - Experiment tracking              │
│  │ GET /                 │  │    │  - Metrics logging (RMSE, MAE, R²)  │
│  │   → Serves UI         │  │    │  - Run history                      │
│  ├───────────────────────┤  │    └─────────────────────────────────────┘
│  │ POST /predict-energy │  │                                           
│  │   → Model inference  │  │    ┌─────────────────────────────────────┐
│  ├───────────────────────┤  │    │   PROMETHEUS (Port 9090)            │
│  │ GET /predictions      │  │    │  - Scrapes /metrics from API       │
│  │   → Query DB          │  │    │  - Time-series database            │
│  ├───────────────────────┤  │    └─────────────────┬───────────────────┘
│  │ GET /metrics          │  │                      │
│  │   → Prometheus        │  │                      ▼
│  └───────────┬───────────┘  │    ┌─────────────────────────────────────┐
│              │              │    │   GRAFANA (Port 3000)                 │
│              │              │    │  - Dashboards                         │
│              ▼              │    │  - Visualizations                    │
│  ┌───────────────────┐      │    └─────────────────────────────────────┘
│  │  PostgreSQL        │◄─────┘
│  │  (Port 5432)       │
│  │  - prediction logs │
│  └───────────────────┘
│              ▲
│              │
└──────────────┼────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     TRAINER (One-time job)                               │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────┐    │
│  │ preprocess.py       │─▶│ train.py             │─▶│ model.pkl     │    │
│  │ - Load raw data     │  │ - Train model        │  │ - Saved model │    │
│  │ - Feature eng.      │  │ - Log to MLflow      │  │ - Shared vol  │    │
│  └─────────────────────┘  └─────────────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

──────────────────────────────────────────────────────────────────────────
                         JENKINS PIPELINE
──────────────────────────────────────────────────────────────────────────
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Checkout │─▶│Start MLflow│─▶│  Build   │─▶│ Training │─▶│ Deploy   │
│   Code   │  │ Service  │  │ Images   │  │  Model   │  │   All    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Programming | Python | 3.11 | ML model, API |
| ML Framework | scikit-learn | Latest | LinearRegression pipeline |
| Experiment Tracking | MLflow | 2.12.1 | Log metrics, params |
| API Framework | FastAPI | Latest | REST API |
| Web Server | Uvicorn | Latest | ASGI server |
| Database | PostgreSQL | 15 | Prediction logging |
| Reverse Proxy | Nginx | Alpine | Port 80 routing |
| Monitoring | Prometheus | 2.51.0 | Metrics collection |
| Visualization | Grafana | 10.4.0 | Dashboards |
| CI/CD | Jenkins | LTS | Automation pipeline |
| Containerization | Docker Compose | 3.9 | Orchestration |

## Prerequisites

Before running this project, ensure you have:

1. **Docker Desktop** installed and running
   - Download: https://www.docker.com/products/docker-desktop/
   - Enable "Expose daemon on tcp://localhost:2375" in Docker Desktop settings

2. **Jenkins** installed locally (for CI/CD)
   - Download: https://www.jenkins.io/download/
   - Requires Java 17+

3. **GitHub Account** with Personal Access Token
   - Go to: Settings → Developer settings → Personal access tokens
   - Generate token with `repo` scope

4. **UCI Dataset** (manually downloaded)
   - URL: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
   - Place at: `data/raw/household_power_consumption.txt`

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/energy-mlops-project.git
cd energy-mlops-project
```

### Step 2: Download Dataset

Download the UCI dataset and place it in:
```
data/raw/household_power_consumption.txt
```

### Step 3: Run with Docker Compose

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps
```

### Step 4: Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard UI** | http://localhost | Main web interface |
| API Docs | http://localhost:8000/docs | FastAPI interactive docs |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics explorer |

### Step 5: Make a Prediction

Using the UI at http://localhost:
1. Enter Hour (0-23)
2. Select Day of Week
3. Click "Predict"

Or using curl:
```bash
curl -X POST http://localhost/predict-energy \
  -H "Content-Type: application/json" \
  -d '{"hour_of_day": 8, "day_of_week": 1}'
```

## Project Structure

```
energy-mlops-project/
├── data/
│   ├── raw/                      # Raw dataset (not in git)
│   │   └── household_power_consumption.txt
│   └── processed/                # Processed features
│       └── features.csv
├── src/
│   ├── preprocess.py              # Data preprocessing
│   └── train.py                  # Model training with MLflow
├── api/
│   ├── main.py                   # FastAPI application
│   └── static/
│       └── index.html            # Dashboard UI
├── models/
│   └── model.pkl                 # Trained model
├── mlruns/                       # MLflow tracking data
├── monitoring/
│   ├── prometheus.yml            # Prometheus config
│   └── grafana/
│       ├── provisioning/         # Grafana auto-provisioning
│       └── dashboards/           # Dashboard JSON
├── nginx/
│   └── nginx.conf                # Reverse proxy config
├── docker-compose.yml            # All services orchestration
├── Dockerfile.api                # API container
├── Dockerfile.trainer            # Training container
├── Jenkinsfile                   # CI/CD pipeline
├── requirements.txt              # Training dependencies
├── requirements.api.txt         # API dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── CONCEPTS.md                   # Detailed explanation of concepts
```

## Understanding Each Component

### 1. Data Preprocessing (`src/preprocess.py`)

This script transforms raw UCI power consumption data into machine-learning-ready features.

**What it does:**
1. Loads the raw semicolon-separated file
2. Handles missing values ("?" characters)
3. Extracts time-based features (hour, day, weekend, peak)
4. Saves cleaned data to CSV

**Input**: `data/raw/household_power_consumption.txt`

**Output**: `data/processed/features.csv`

**Features extracted:**
- `hour_of_day`: Hour of the day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: 1 if Saturday/Sunday, 0 otherwise
- `is_peak_hour`: 1 if hour is 7-9am or 5-8pm
- `target`: Global active power consumption

**Run separately:**
```bash
docker compose run --rm trainer python src/preprocess.py
```

### 2. Model Training (`src/train.py`)

Trains a scikit-learn pipeline and logs metrics to MLflow.

**Pipeline Structure:**
```python
Pipeline([
    ('scaler', StandardScaler()),   # Normalize features
    ('model', LinearRegression())  # Train linear model
])
```

**MLflow Tracking:**
- Parameters logged: model_type, features, train_size
- Metrics logged: RMSE, MAE, R²
- Location: http://localhost:5000

**Training Process:**
1. Load processed features from CSV
2. Split into 80% train, 20% test
3. Fit pipeline on training data
4. Evaluate on test data
5. Log metrics to MLflow
6. Save model to shared volume

**Output:** `models/model.pkl`

**Run separately:**
```bash
# Start MLflow first
docker compose up -d mlflow

# Then run training
docker compose run --rm trainer python src/train.py
```

### 3. FastAPI Application (`api/main.py`)

The API handles all HTTP requests and integrates with the database.

**Key Features:**
- Serves HTML dashboard
- Runs model inference
- Logs predictions to PostgreSQL
- Exposes Prometheus metrics
- Handles model reloading

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Serve HTML dashboard |
| POST | /predict-energy | Run prediction, log to DB |
| GET | /predictions | Get last 10 predictions |
| GET | /stats | Total predictions + model status |
| GET | /health | Health check |
| GET | /metrics | Prometheus scrape endpoint |
| POST | /reload-model | Reload model without restart |

**Database Integration:**
- Uses SQLAlchemy core (not ORM)
- Creates `predictions` table on startup
- Logs every prediction with timestamp, input features, and output

**Prometheus Integration:**
- Uses `prometheus-fastapi-instrumentator`
- Automatically tracks:
  - Request count by handler/status
  - Request duration histogram
  - Response size

### 4. Web Dashboard (`api/static/index.html`)

Modern dark-themed dashboard built with pure HTML/CSS/JavaScript.

**Features:**
- **Prediction Form**: Input hour and day of week
- **Auto-computed Badges**: Weekend and Peak Hour indicators update instantly
- **Result Display**: Animated prediction result
- **System Status**: Total predictions and model status
- **Quick Links**: Buttons to MLflow, Grafana, Prometheus
- **Recent Predictions Table**: Auto-refreshes every 10 seconds
- **Status Indicator**: Shows if model is ready

**Design:**
- Dark theme with blue accent colors
- JetBrains Mono for numbers
- IBM Plex Sans for text
- Responsive layout

### 5. Docker Compose

Orchestrates 7 services working together.

**Services:**

| Service | Image | Purpose | Port |
|---------|-------|---------|------|
| postgres | postgres:15-alpine | Database | 5432 |
| mlflow | ghcr.io/mlflow/mlflow:v2.12.1 | Experiment tracking | 5000 |
| trainer | energy-mlops-trainer (built) | Training job | - |
| api | energy-mlops-api (built) | FastAPI app | 8000 |
| nginx | nginx:alpine | Reverse proxy | 80 |
| prometheus | prom/prometheus:v2.51.0 | Metrics | 9090 |
| grafana | grafana/grafana:10.4.0 | Dashboards | 3000 |

**Volume Mounts:**
- `./data` → `/app/data` (raw and processed data)
- `./models` → `/app/models` (trained model)
- `./mlruns` → `/mlruns` (MLflow tracking)

### 6. Jenkins Pipeline

Automated CI/CD that builds, trains, and deploys the entire system.

**Pipeline Stages:**

```groovy
stage("Checkout")           // Pull code from GitHub
stage("Start MLflow")       // Start MLflow container, wait 15s
stage("Build Docker Images")// Build trainer & API images
stage("Run Training")       // Train model, save to volume
stage("Deploy All Services")// Start all containers
```

**Why this pipeline?**
1. **Checkout**: Gets latest code from GitHub
2. **Start MLflow**: Ensures MLflow is running before training
3. **Build**: Creates Docker images locally
4. **Training**: Runs the training job (one-time)
5. **Deploy**: Starts all services

**Configuration:**
- Runs on any agent (`agent any`)
- 30-minute timeout
- Uses Windows batch commands (`bat`)
- Uses PowerShell for sleep (`Start-Sleep`)

## API Reference

### Prediction Endpoint

Make a prediction with time-based features.

```bash
POST /predict-energy
Content-Type: application/json

# Minimal request
{"hour_of_day": 8, "day_of_week": 1}

# Full request (optional fields auto-computed)
{"hour_of_day": 20, "day_of_week": 5, "is_weekend": 1, "is_peak_hour": 1}
```

**Response:**
```json
{
  "predicted_energy_consumption": 1.234,
  "unit": "kW",
  "timestamp": "2026-04-10T14:30:00.123456"
}
```

### Get Predictions

Retrieve the last 10 predictions.

```bash
GET /predictions
```

**Response:**
```json
{
  "predictions": [
    {
      "id": 1,
      "timestamp": "2026-04-10 14:30:00",
      "hour_of_day": 8,
      "day_of_week": 1,
      "is_weekend": 0,
      "is_peak_hour": 1,
      "prediction": 1.234
    }
  ]
}
```

### Get Stats

Get system statistics.

```bash
GET /stats
```

**Response:**
```json
{
  "total_predictions": 42,
  "model_loaded": true
}
```

### Health Check

Check if API is running.

```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Monitoring Guide

### Prometheus Metrics

The API automatically exposes metrics at `/metrics`.

**Available Metrics:**
- `http_requests_total` - Total requests by handler, method, status
- `http_request_duration_seconds` - Request duration histogram
- `http_response_size_bytes` - Response size histogram

### Prometheus Queries

1. **Request Rate** (requests per second):
   ```
   rate(http_requests_total[1m])
   ```

2. **P95 Latency** (seconds):
   ```
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
   ```

3. **Total Predictions (24h)**:
   ```
   increase(http_requests_total{handler="/predict-energy"}[24h])
   ```

### Grafana Dashboard

1. Login: http://localhost:3000
   - Username: `admin`
   - Password: `admin`

2. Import dashboard:
   - Go to Dashboards → Import
   - Upload: `monitoring/grafana/dashboards/energy_dashboard.json`

3. Or create manually:
   - Dashboard → New → Panel
   - Use Prometheus queries above

## Common Commands

### Container Management

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes database)
docker compose down -v

# View logs for specific service
docker compose logs -f api
docker compose logs -f trainer

# Rebuild and restart
docker compose up -d --build

# Check status
docker compose ps
```

### Database Queries

```bash
# Query last 10 predictions
docker compose exec postgres psql -U user -d energydb -c "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;"

# Count total predictions
docker compose exec postgres psql -U user -d energydb -c "SELECT COUNT(*) FROM predictions;"

# Delete all predictions (reset)
docker compose exec postgres psql -U user -d energydb -c "DELETE FROM predictions;"
```

### Generate Test Traffic

```powershell
# Make 20 predictions using PowerShell
1..20 | ForEach-Object { 
    Invoke-RestMethod -Uri 'http://localhost/predict-energy' -Method POST -ContentType 'application/json' -Body '{"hour_of_day":8,"day_of_week":1}' 
}
```

### Run Individual Components

```bash
# Run preprocessing only
docker compose run --rm trainer python src/preprocess.py

# Run training only
docker compose run --rm trainer python src/train.py

# Start only MLflow
docker compose up -d mlflow

# Start only API
docker compose up -d api
```

## Troubleshooting

### API Returns 503 "Model not loaded yet"

**Cause**: The model hasn't been trained yet

**Solution**:
```bash
# Run training
docker compose run --rm trainer python src/train.py

# Or check if model exists
dir models\
```

### Port Already in Use

**Cause**: Another service using port 80, 5432, 5000, 8000, 9090, or 3000

**Solution**:
```powershell
# Find what's using port 80
netstat -ano | findstr :80

# Stop the conflicting service or change the port in docker-compose.yml
```

### Database Connection Error

**Cause**: PostgreSQL not ready or connection string wrong

**Solution**:
```bash
# Wait for healthcheck
docker compose ps

# Check database logs
docker compose logs postgres
```

### Grafana Shows No Data

**Cause**: No metrics collected yet

**Solution**:
```bash
# Make some predictions
curl -X POST http://localhost/predict-energy -H "Content-Type: application/json" -d '{"hour_of_day":8,"day_of_week":1}'

# Wait 15 seconds for Prometheus to scrape
# Then refresh Grafana
```

### Jenkins Pipeline Fails

**Common causes:**
1. Docker not in PATH - Add Docker to system PATH
2. GitHub credentials wrong - Re-add credentials in Jenkins
3. Port conflicts - Stop other services before running pipeline

## Jenkins Setup (For Team Members)

If you're setting up Jenkins for the first time:

### Step 1: Install Jenkins

1. Download from https://www.jenkins.io/download/
2. Run installer, accept defaults
3. Open http://localhost:8080
4. Enter initial admin password (found in `C:\ProgramData\Jenkins\.jenkins\secrets\initialAdminPassword`)
5. Install suggested plugins
6. Create admin user

### Step 2: Configure Docker

1. Open Docker Desktop → Settings → General
2. Check "Expose daemon on tcp://localhost:2375 without TLS"
3. Click Apply & Restart

### Step 3: Create Pipeline Job

1. Open http://localhost:8080
2. Click "New Item"
3. Enter name: `energy-mlops`
4. Select "Pipeline" → Click OK
5. In Pipeline section:
   - Definition: "Pipeline script from SCM"
   - SCM: "Git"
   - Repository URL: `https://github.com/YOUR_USERNAME/energy-mlops-project.git`
   - Credentials: Add → Username with password → Enter GitHub username + Personal Access Token
   - Branch: `*/main`
   - Script Path: `Jenkinsfile`
6. Click Save
7. Click "Build Now"

### Step 4: Monitor Pipeline

Watch the console output to see each stage execute. All green checkmarks = success!

## Learning Outcomes

After studying this project, you will understand:

- ✅ Data preprocessing with pandas
- ✅ Feature engineering for time-series ML
- ✅ scikit-learn pipelines (scaler + model)
- ✅ MLflow experiment tracking
- ✅ FastAPI REST API development
- ✅ PostgreSQL database integration
- ✅ Prometheus metrics collection
- ✅ Grafana dashboard creation
- ✅ Docker Compose orchestration
- ✅ Jenkins CI/CD pipelines
- ✅ Reverse proxy with Nginx

## How Each File Contributes

| File | Purpose |
|------|---------|
| `src/preprocess.py` | Data cleaning and feature extraction |
| `src/train.py` | Model training with MLflow logging |
| `api/main.py` | FastAPI app with all endpoints |
| `api/static/index.html` | Web dashboard UI |
| `docker-compose.yml` | All 7 services orchestration |
| `Dockerfile.api` | API container definition |
| `Dockerfile.trainer` | Training container definition |
| `Jenkinsfile` | CI/CD pipeline definition |
| `monitoring/prometheus.yml` | Prometheus scrape configuration |
| `monitoring/grafana/dashboards/*.json` | Grafana dashboard |
| `nginx/nginx.conf` | Reverse proxy configuration |

## Service URLs Summary

| Service | Port | URL | Login |
|---------|------|-----|-------|
| Dashboard UI | 80 | http://localhost | - |
| API | 8000 | http://localhost:8000 | - |
| API Docs | 8000 | http://localhost:8000/docs | - |
| MLflow | 5000 | http://localhost:5000 | - |
| Grafana | 3000 | http://localhost:3000 | admin/admin |
| Prometheus | 9090 | http://localhost:9090 | - |
| PostgreSQL | 5432 | localhost:5432 | user/password |

## License

This project is for educational purposes. The UCI dataset has its own license from UCI Machine Learning Repository.

## Author

Created as a complete MLOps demonstration project showcasing modern machine learning deployment practices using industry-standard tools.
