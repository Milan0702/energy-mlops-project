# Smart Energy Consumption Prediction System

An end-to-end MLOps demonstration project that trains a linear regression model on household power consumption data and serves predictions through a REST API with a web dashboard. Includes experiment tracking, monitoring, and CI/CD automation.

## Architecture

```
  Browser
     │
     ▼
  Nginx (:80)  ←─── serves UI + proxies API
     │
     ▼
  FastAPI (:8000)
  ├── GET  /          → HTML dashboard UI
  ├── POST /predict-energy → sklearn model inference
  ├── GET  /predictions    → last 10 from PostgreSQL
  ├── GET  /stats          → total count + model status
  └── GET  /metrics        → Prometheus scrape target
     │
     ├──→ PostgreSQL (:5432)  [logs every input + prediction]
     └──→ model.pkl           [scikit-learn LinearRegression Pipeline]

  train.py → MLflow (:5000) [logs RMSE, MAE, R2]
  train.py → model.pkl      [saved to shared volume]

  FastAPI /metrics → Prometheus (:9090) → Grafana (:3000)

  Jenkins (local PC) → GitHub → docker compose up
```

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| UI | HTML + CSS + JS | Dashboard — predict, view logs, system status |
| ML Model | scikit-learn | LinearRegression pipeline, simple + explainable |
| Experiment Track | MLflow | Logs RMSE, MAE, R2 per training run |
| Prediction API | FastAPI | REST API + serves the UI |
| Database | PostgreSQL | Logs every prediction request (input + output) |
| Reverse Proxy | Nginx | Single entry point on port 80 |
| Monitoring | Prometheus + Grafana | Request rate, response time, prediction count |
| CI/CD | Jenkins (local) | Build, train, deploy pipeline on demand |
| Orchestration | Docker Compose | Runs all services with one command |

## Prerequisites

- Docker Desktop installed and running
- Jenkins installed locally (see setup guide at bottom of this README)
- UCI dataset:
  Download: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
  Place at:  data/raw/household_power_consumption.txt

## Quick Start

1. Clone the repository:
   git clone https://github.com/yourusername/energy-mlops-project.git
   cd energy-mlops-project

2. Download dataset and place at data/raw/household_power_consumption.txt

3. Run preprocessing:
   docker compose run --rm trainer python src/preprocess.py

4. Run training:
   docker compose run --rm trainer python src/train.py

5. Start all services:
   docker compose up -d

6. Open the UI:
   http://localhost

## Service URLs

| Service | URL | Notes |
|---|---|---|
| Dashboard UI | http://localhost | Main entry point — use this for demo |
| API (direct) | http://localhost:8000 | FastAPI docs at /docs |
| MLflow UI | http://localhost:5000 | View training experiments and metrics |
| Grafana | http://localhost:3000 | Login: admin / admin |
| Prometheus | http://localhost:9090 | Raw metrics query interface |
| PostgreSQL | localhost:5432 | DB: energydb, user: user, pass: password |

## API Reference

POST /predict-energy

Minimal request:
  curl -X POST http://localhost/predict-energy \
    -H "Content-Type: application/json" \
    -d '{"hour_of_day": 8, "day_of_week": 1}'

Full request (all fields):
  curl -X POST http://localhost/predict-energy \
    -H "Content-Type: application/json" \
    -d '{"hour_of_day": 20, "day_of_week": 5, "is_weekend": 1, "is_peak_hour": 1}'

Response:
  {"predicted_energy_consumption": 1.23, "unit": "kW", "timestamp": "2024-01-15T08:00:00"}

## Demo Script (for presentations)

0. Open http://localhost → show the dark dashboard UI
1. Fill in: Hour = 8, Day = Monday → watch Weekend/Peak badges update live
2. Click Predict → result animates in, table updates automatically
3. Make 5–10 more predictions to generate data
4. Show the Recent Predictions table filling up
5. Click "MLflow UI" link on the dashboard → show training metrics (RMSE, R2)
6. Click "Grafana" link → show request rate chart going up
7. Show Prometheus at localhost:9090 → query http_requests_total
8. Run: docker exec -it <postgres-container> psql -U user -d energydb -c "SELECT * FROM predictions LIMIT 5;"
9. Show Jenkins pipeline stages (if configured)

## Useful Commands

docker compose ps                          # check all services
docker compose logs api                    # view API logs
docker compose logs trainer                # view training logs
docker compose run --rm trainer            # re-run training
docker compose down                        # stop everything
docker compose down -v                     # stop + delete database

# Query predictions in PostgreSQL
docker exec -it $(docker compose ps -q postgres) \
  psql -U user -d energydb -c "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;"

# Generate traffic for monitoring demo
for i in {1..20}; do curl -s -X POST http://localhost/predict-energy \
  -H "Content-Type: application/json" \
  -d '{"hour_of_day":8,"day_of_week":1}' > /dev/null; done

## Troubleshooting

model.pkl not found / API returns 503
  → Trainer hasn't finished. Run: docker compose run --rm trainer
  → Or check: docker compose logs trainer

Port 80 already in use
  → Stop the process using port 80, or change nginx port in docker-compose.yml

API not responding
  → Check: docker compose logs api
  → Check: docker compose ps (all services should show as running)

Grafana shows no data
  → Make some predictions first, then wait 15 seconds for Prometheus to scrape

PostgreSQL connection error
  → Wait 10–15 seconds after docker compose up for the healthcheck to pass
