# End-to-End MLOps System for Churn Prediction


## Introduction

This repository contains a production-oriented end-to-end MLOps system for churn prediction in a bank. The project demonstrates core MLOps principles, including orchestration, model deployment, model versioning, experiment tracking, monitoring, automated retraining triggered by data drift, best practices and much more. 

**The core data science documentation with all context and methodology including exploratory data analysis, modeling and hyperparameter tuning is provided in the research's [Experiment.ipynb](./research/Experiment.ipynb).**

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [ML Lifecycle Design](#ml-lifecycle-design)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Sample Run](#sample-run)
- [Limitations & Future Improvements](#limitations--future-improvements)


## Project Overview

### Problem Statement

Customer retention plays a critical role in maintaining long-term profitability in the modern banking environments. By predicting which customers are likely to leave in advance, companies can activate retention strategies. This project builds a predictive model to flag potential churners for a bank’s credit card services, enabling the bank to take targeted retention actions.

### High-level solution





## System Architecture
![System Architecture](./docs/assets/Churn_Project_Architecture.svg)



---
#### 1. Orchestration

**Prefect** is a code-based orchestration tool that acts as the backbone of the training and monitoring workflows.
- The **training pipeline** orchestrates data ingestion, data validation, data transformation, model training, model evaluation and model pushing.
- The **data monitoring pipeline** runs independently. It consumes prediction data stored in an S3 Bucket and performs a statistical test called Kolmogorov-Smirnov to detect data drift. Comparing the new coming data with a reference dataset, the monitoring pipeline triggers the training pipeline when drift thresholds are exceeded. Alongside, it generates and stores a data monitoring report using Evidently for diagnostics and visualization.

Prefect enables scheduled monitoring with deployed flows, which are registered and runnable versions of the pipelines.

---

#### 2. Experiment Tracking and Model Versioning

**MLflow** is used for experiment tracking and model versioning.
- All experiments, metrics, parameters, and model artifacts are tracked in MLflow
- The selected “best” (champion) model is persisted to a dedicated S3 location
- Inference services load models directly from S3.

---

#### 3. Hyperparameter Tuning

The training pipeline is feeded with optimized hyperparameters reached in the bayesian search of **Optuna** (check in the [experiment](./research/Experiment.ipynb)). MLflow logs a child run at each trial of hyperparameter combination targeting the improvement of a metric value. The best combination is provided in the parent run when the study is finished. The best combination of hyperparameters is passed to the training pipeline configuration in the params.yaml file.

---

#### 4. Serving & Inference Layer

The serving stack runs on a single **Amazon EC2** instance using Docker Compose.
- **FastAPI** exposes the prediction endpoints
- At application startup, the FastAPI service loads the latest champion model from S3 into memory
- **Streamlit** provides a lightweight frontend for interaction and demonstration
- Both services run in isolated **Docker** containers with registered images from **Amazon ECR**.
- The FastAPI service exposes a metrics endpoint that is scraped by **Prometheus** to perform monitoring and alerting.
- **Grafana** is used for the better visualization of metrics.

---

#### 5. CI/CD & Deployment Strategy

**GitHub Actions** is responsible for continuous integration and deployment:

**CI:**
- Linting and unit tests
- Docker image builds for API and frontend
- Pushes images to Amazon ECR

**CD:**
- Secure SSH connection to the EC2 instance
- Pulls updated images from ECR
- Restarts services via Docker Compose

AWS IAM roles are used instead of long-lived credentials for EC2, aligning with security best practices.


---

## ML Lifecycle Design

This project implements a **fully orchestrated, reproducible, and production-oriented machine learning lifecycle**, covering the complete path from raw data ingestion to automated model promotion and deployment. The lifecycle is designed around clear separation of concerns, deterministic artifacts, and explicit decision gates to prevent unsafe or noisy model updates.

The ML lifecycle consists of the following stages:

**Ingestion → Validation → Transformation → Training → Evaluation → Promotion → Deployment**

Each stage produces a well-defined artifact that is consumed by downstream stages, ensuring traceability, reproducibility, and failure isolation.

Each ML lifecycle stage was implemented in the following order:
- [config](./config)
- [entities](./src/churn_project/entity)
- [configuration manager](./src/churn_project/config/configuration.py)
- [components](./src/churn_project/components)
- [pipeline](./src/churn_project/orchestrator/training_flow.py)

---

#### 1. Data Ingestion

**Objective:** Extract raw data from the source system and produce training and testing datasets.

**Design:**

* Data is extracted directly from a relational database using SQLAlchemy.
* A deterministic train/test split is applied using a fixed random seed.

---

#### 2. Data Validation

**Objective:** Enforce schema integrity and data sanity before allowing any training to proceed.

**Validation Checks:**

* Column presence validation against the declared schema
* Data type validation per column
* Missing value detection

---

#### 3. Data Transformation & Feature Engineering

**Objective:** Convert validated raw data into model-ready numerical representations while preserving reproducibility.

**Key Steps:**

* Feature engineering using a custom transformer:
  * Behavioral ratios (e.g., activity growth)
  * Customer value aggregation features
* Feature scaling using `StandardScaler`
* Class imbalance handling using **SMOTE**, applied **only to training data**

---

#### 4. Model Training

**Objective:** Train a candidate model using validated, transformed data and register it with full metadata.

**Model Strategy:**

* The model type is selected via configuration (e.g., XGBoost or Random Forest).
* Hyperparameters are defined externally and logged in config's params.yaml.
* Training occurs only on resampled, transformed data.

**MLflow Integration:**

* Training parameters and metrics are logged
* A complete inference pipeline (preprocessor + model) is registered in the MLflow Model Registry

Training only produces a **candidate**.

---

#### 5. Model Evaluation & Acceptance

**Objective:** Decide whether a newly trained model is eligible for production.

**Evaluation Logic:**

* The current production (champion) model is loaded if it exists
* Both models are evaluated on the same untouched test dataset
* ROC-AUC is used as the primary selection metric

**Acceptance Rule:**

* A model is accepted **only if** it exceeds the production model’s AUC by a defined margin
* If no production model exists, the new model is accepted by default
* Each model version is explicitly tagged as `approved` or `rejected`

---

#### 6. Model Promotion & Production Deployment

**Objective:** Safely promote an approved model and make it available for inference.

**Promotion Steps:**

1. Approved models are copied to a dedicated production registry
2. The promoted version is assigned the alias `champion`
3. Model artifacts and metadata are exported and uploaded to a production S3 location

---

## Project Structure

```
churn-project/
├── .github/workflows/          # CI/CD pipelines (GitHub Actions)
│
├── config/                     # Centralized YAML configuration
│   ├── config.yaml             # Training Pipeline configurations
│   ├── params.yaml             # Model and training hyperparameters
│   └── schema.yaml             # Input data schema
│
├── docs/                       # Documentation content
│
├── frontend/                   # User-facing application
│   ├── Dockerfile              # Container for Streamlit frontend
│   └── streamlit_app.py        # Interactive UI for predictions and insights
│
├── monitoring/                 # Observability, drift detection, and retraining
│   ├── data_drift/
│   │   ├── monitoring_flow.py     # Data monitoring Prefect flow
│   │   ├── monitoring_tasks.py    # Data monitoring Prefect tasks
│   │   └── retraining_trigger.py  # Automated retraining trigger logic
│   ├── grafana/
│   │   └── dashboard.json      # Grafana dashboard configuration
│   └── prometheus/
│       ├── prometheus.yml      # Metrics scraping configuration
│       └── alert_rules.yml     # Alerting rules
│
├── research/                   # Exploratory and experimental work
│   ├── data_drift_study.ipynb  # Drift analysis experiment
│   └── Experiment.ipynb        # Core Data Science Context and Methodology 
│
├── src/churn_project/          # Core application and ML logic
│   ├── api/                    # Inference API
│   │   ├── app.py              # FastAPI application
│   │   ├── Dockerfile          # Production inference container
│   │   └── schemas.py          # Request/response schemas
│   │
│   ├── aws/                       # AWS integrations
│   │   ├── s3_utils.py            # S3 utilities (model and artifact loading)
│   │   └── monitoring_logging.py  # Centralized logging to AWS/S3 for API usage
│   │
│   ├── components/             # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── config/                 # Runtime configuration management
│   │   └── configuration.py
│   │
│   ├── constants/              # Global constants
│   │
│   ├── entity/                 # Typed configuration and artifact entities
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   │
│   ├── inference/              # Prediction logic
│   │   └── prediction_service.py
│   │
│   ├── orchestrator/           # Training orchestration
│   │   └── training_flow.py    # Prefect training pipeline flow
│   │
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Centralized logging
│   └── utils.py                # Shared utilities
│
├── tests/                      # Automated test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── conftest.py             # Pytest fixtures
│
├── .env.example                # Environment variable template
├── .pre-commit-config.yaml     # Code quality hooks
├── README.md                   # High-level project documentation
├── docker-compose.prod.yaml    # Production orchestration
├── docker-compose.yaml         # Local development orchestration
├── main.py                     # Training pipeline entry point
├── poetry.lock                 # Locked dependencies
└── pyproject.toml              # Dependency and project configuration (Poetry)

```
---

## Setup

#### Prerequisites

- Python 3.12
- Poetry
- Docker and Docker Compose
- MySQL
- AWS account
- Recommended: Unix-based system (MacOS/Linux)

### Setup Instructions

Follow these steps to set up the project environment and run the project on your local machine.

### 1. Initial Setup

**1.1. Clone the Repository**

```bash
git clone https://github.com/Pedro-Vital/churn-project.git
cd churn-project
```

**1.2. Ensure Poetry is installed**

Check whether Poetry is already available:
```bash
poetry --version
```
If not installed, install it using the official installer:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Then restart your terminal or ensure Poetry is in your PATH.

**1.3. Configure Poetry to use the project-local virtual environment (recommended)**
```bash
poetry config virtualenvs.in-project true
```
**1.4. Install project dependencies**
```bash
poetry install
```
**1.5. Activate the virtual environment**
```bash
poetry env activate
```
If it wasn't activated, restart your terminal.

**1.6. Set up environment variables**

Create your local environment file:
```bash
cp .env.example .env
```
Edit the .env file and provide the necessary values ​​when you obtain them.

**1.7. Verify the setup**

You can perform a quick sanity check:
```bash
python -c "import churn_project; print('Environment OK')"
```

If no errors occur, the environment is correctly configured.

### 2. Database Setup

**2.1. Download the dataset from Kaggle:**
[Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

**2.2. Add the dataset to MySQL**

- Remove the "naive" columns
- Insert the dataset in a table named `churners` 

**2.3. Add database credentials to .env**

### 3. MLflow Setup

**3.1 Start the MLflow server runinng:**
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

**3.2 Access MLflow UI: `http://127.0.0.1:5000`**

### 4. Prefect Setup (Orchestration)

**4.1 Start the Prefect server running:**

```bash
prefect server start
```

**4.2 Access the UI: `http://127.0.0.1:4200`**

**4.3 Create a work pool**

```bash
prefect work-pool create churn-pool --type process
```

**4.4 Deploy flows**

Training pipeline

```bash
prefect deploy src/churn_project/orchestrator/training_flow.py:training_flow \
  -n churn-train

```

Data monitoring pipeline

```bash
prefect deploy monitoring/data_drift/monitoring_flow.py:data_monitoring_flow \
  -n monitoring

```

During deployment, Prefect may ask configuration questions. Default answers are sufficient.

**4.5 Start a worker**

```bash
prefect worker start --pool churn-pool

```
- **Multiple terminals are recommended**:
    - Terminal 1: MLflow server
    - Terminal 2: Prefect server
    - Terminal 3: Prefect worker
    - Terminal 4 (optional): Training / inference commands

### 5. S3 Bucket Setup

**5.1. Create an S3 bucket named churn-production**

**5.2 Create an IAM User with the following policy: `AmazonS3FullAccess`**

**The policy hereby used is useful just to reduce IAM complexity. In a real-world case, we would use a custom least-privilege policy.**

**5.3. Add AWS credentials to .env**

***Now you're all set to run the project locally. But if you want to run it completely using an EC2 instance, proceed with the setup.***

### 6. Containers Setup 

**6.1 Create two ECR Repositories: `churn-fastapi` and `churn-streamlit`**\
Configuration:
- Visibility: Private
- Image tag mutability: Mutable
- Enable Scan on push
- Default for the rest

**6.2 Click each repository and copy the repository URI**

**6.3 Replace the repository URI and AWS_REGION of [docker-compose.prod.yaml](./docker-compose.prod.yaml) with your actual repository URI and region**

### 7. IAM Setup

**7.1 Create an IAM User with the following policy: `AmazonEC2ContainerRegistryPowerUser`**\
It will be used by GitHub Actions

**7.1 Create an IAM Role named `churn-ec2-role` with the following policies: `AmazonEC2ContainerRegistryReadOnly` and `AmazonS3FullAccess`**\
Configuration:
- Use case: EC2

**The policy hereby used is useful just to reduce IAM complexity. In a real-world case, we would use a custom least-privilege policy.**

### 8. EC2 Bootstrap

**8.1 Launch EC2**
- Name: churn-prod-ec2
- OS: Amazon Linux 2023 (AL2023)
- Instance type: t3.small
- Create or Select Key Pair. You will later store this key as a GitHub Secret.
- VPC: Default
- Subnet: Default
- Auto-assign Public IP: Enable
- Assign it a security group named churn-prod-sg with the following inbound rules:

| Port | Source | Purpose |
| --- | --- | --- |
| 22 | Your IP | SSH |
| 8000 | 0.0.0.0/0 | FastAPI |
| 8501 | 0.0.0.0/0 | Streamlit |
| 9090 | Your IP | Prometheus |
| 3000 | Your IP | Grafana |

- Configure storage: 30 GB, type gp3
- IAM Instance Profile (Advanced Details): churn-ec2-role
- Default / Empty for the rest

**Launch Instance**

**8.1 Install Docker + Compose**
- Connect to the EC2 Instance
- Run the following commands:
```bash
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.25.0/docker-compose-linux-x86_64 \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
```

### 9 GitHub Secrets Setup

**9.1 Add the following as variables**
- AWS_ACCOUNT_ID → Your AWS account ID
- AWS_REGION → Your AWS Region
- EC2_USER → if you didn't change it, it is `ec2-user` (Amazon Linux) 

**9.2 Add the following as secrets**
- AWS_ACCESS_KEY_ID → IAM user access key for ECR auth only (from 7.1)
- AWS_SECRET_ACCESS_KEY → IAM user secret key for ECR auth only (from 7.1)
- EC2_HOST → Your Public IPv4 or DNS
- EC2_SSH_KEY → Private key → That is the key selected at EC2 launch. Use `cat <key-name>` to access, then copy and paste it entirely to secrets.

***Now you can run deployment adding a minimal modification to the repository. It will trigger GitHub Actions workflow.***

---



















## Sample Run



---

## Limitations & Future Improvements

---
# Grafana Dashboard (Import Instructions)

1. Open Grafana
   `http://localhost:3000`

2. Go to
   **Dashboards → Import**

3. Upload or paste the dashboard JSON file from this repository.

4. When Grafana asks for a datasource mapping, set:
   **DS_PROMETHEUS → your Prometheus datasource**

5. Click **Import**.
   The dashboard loads immediately and shows:

   * API request throughput
   * API latency (p50/p95/p99)
   * HTTP error rate
   * Prediction throughput
   * Prediction latency
   * Churn output distribution
   * Process CPU and memory

6. Ensure Prometheus is scraping your FastAPI service and exposing `/metrics`.

Done.

“Model is loaded at application startup”

“New models require redeployment”

---

- The retraining problem

The only events that trigger a reload are:

Container restart

EC2 restart

Uvicorn restart

Crash + restart

New deployment
