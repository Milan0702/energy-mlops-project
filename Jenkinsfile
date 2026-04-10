// Prerequisites:
// - Jenkins installed locally on the developer's PC (not in Docker)
// - Jenkins user has Docker and Docker Compose on PATH
// - Pipeline connected to a GitHub repository
// - Workspace is the project root

pipeline {
    agent any
    
    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
    }

    environment {
        PROJECT_NAME = "energy-mlops"
    }

    stages {
        stage("Checkout") {
            steps {
                checkout scm
                echo "Code checked out from GitHub"
            }
        }

        stage("Start MLflow") {
            steps {
                bat "docker compose -p %PROJECT_NAME% up -d mlflow"
                echo "MLflow started, waiting 15 seconds..."
                powershell "Start-Sleep -Seconds 15"
            }
        }

        stage("Build Docker Images") {
            steps {
                bat "docker compose -p %PROJECT_NAME% build"
                echo "Docker images built"
            }
        }

        stage("Run Training") {
            steps {
                echo "Starting model training..."
                bat "docker compose -p %PROJECT_NAME% run --rm trainer"
                echo "Training complete — model.pkl saved to models/"
            }
        }

        stage("Deploy All Services") {
            steps {
                bat "docker compose -p %PROJECT_NAME% up -d"
                powershell "Start-Sleep -Seconds 20"
                bat "docker compose -p %PROJECT_NAME% ps"
                echo "All services deployed."
                echo "UI:         http://localhost"
                echo "API:        http://localhost/predict-energy"
                echo "MLflow:     http://localhost:5000"
                echo "Grafana:    http://localhost:3000"
                echo "Prometheus: http://localhost:9090"
            }
        }
    }

    post {
        failure {
            echo "Pipeline failed. Run: docker compose logs to debug."
        }
        success {
            echo "Pipeline complete. Open http://localhost to use the UI."
        }
    }
}
