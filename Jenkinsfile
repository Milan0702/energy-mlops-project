// Prerequisites:
// - Jenkins installed locally on the developer's PC (not in Docker)
// - Jenkins user has Docker and Docker Compose on PATH
// - Pipeline connected to a GitHub repository
// - Workspace is the project root

pipeline {
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

        stage("Build Docker Images") {
            steps {
                sh "docker compose -p ${PROJECT_NAME} build"
                echo "Docker images built"
            }
        }

        stage("Run Training") {
            steps {
                echo "Starting model training..."
                sh "docker compose -p ${PROJECT_NAME} run --rm trainer"
                echo "Training complete — model.pkl saved to models/"
            }
        }

        stage("Deploy All Services") {
            steps {
                sh "docker compose -p ${PROJECT_NAME} up -d"
                sh "sleep 20"
                sh "docker compose -p ${PROJECT_NAME} ps"
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
