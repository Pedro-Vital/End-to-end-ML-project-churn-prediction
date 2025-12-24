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




---

## Sample Run



---

## Limitations & Future Improvements


## 🔧 Configuration & Secrets Handling

This project uses a **simple and safe configuration pattern**:

* **`config.yaml.example`** → committed to the repository
* **`config.yaml`** → ignored by Git and contains real credentials
* **Environment variables** → optional and override values in `config.yaml`

This prevents leaking database credentials on GitHub while keeping local setup straightforward.

---

## 📁 Step 1 — Create your `config.yaml`

Start by copying the example file:

```bash
cp config.yaml.example config.yaml
```

Inside `config.yaml`, you will see:

```yaml
db_host: null
db_user: null
db_password: null
db_name: null
```

You have two ways to provide your database credentials:

---

### **🔹 Option A — Fill the values directly**

```yaml
db_host: "localhost"
db_user: "root"
db_password: "12345"
db_name: "bank_db"
```

This is the simplest approach for local development.

---

### **🔹 Option B — Use environment variables (preferred for CI)**

```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=12345
export DB_NAME=bank_db
```

These values automatically override whatever is in `config.yaml`.

---

config (config.yaml, schema.yaml, params.yaml)
entities (config and artifact)
configuration manager in src config
components
pipeline


Here is a **straightforward, no-nonsense README section** you can drop directly into your repository.

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

You have to remove "naive" columns











- The retraining problem

The only events that trigger a reload are:

Container restart

EC2 restart

Uvicorn restart

Crash + restart

New deployment
