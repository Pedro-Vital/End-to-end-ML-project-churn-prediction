# End-to-End MLOps System for Churn Prediction

---

## Introduction

This repository contains a production-oriented end-to-end MLOps system for churn prediction in a bank. The project demonstrates core MLOps principles, including orchestration, model deployment, model versioning, experiment tracking, monitoring, automated retraining triggered by data drift, best practices and much more. The core data science context and methodology including exploratory data analysis, modeling and the hyperparameter optimization needed to feed the training pipeline is provided in the research's [Experiment.ipynb](./research/Experiment.ipynb).

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [ML Lifecycle Design](#ml-lifecycle-design)
- [Serving & Inference Strategy](#serving--inference-strategy)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## Project Overview

### Problem Statement

Customer retention plays a critical role in maintaining long-term profitability in the modern banking environments. By predicting which customers are likely to leave in advance, companies can activate retention strategies. This project builds a predictive model to flag potential churners for a bank’s credit card services, enabling the bank to take targeted retention actions.

### High-level solution



---

## System Architecture
![System Architecture](./docs/assets/Churn_Project_Architecture.svg)



---

### Orchestration

**Prefect** is a code-based orchestration tool that acts as the backbone of the training and monitoring workflows.
- The **training pipeline** orchestrates data ingestion, data validation, data transformation, model training, model evaluation and model pushing.
- The **data monitoring pipeline** runs independently. It consumes prediction data stored in an S3 Bucket and performs a statistical test called Kolmogorov-Smirnov to detect data drift. Comparing the new coming data with a reference dataset, the monitoring pipeline triggers the training pipeline when drift thresholds are exceeded. Alongside, it generates and stores a data monitoring report using Evidently for diagnostics and visualization.

Prefect enables scheduled monitoring with deployed flows, which are registered and runnable versions of the pipelines.

### Experiment Tracking and Model Versioning

**MLflow** is used for experiment tracking and model versioning.
- All experiments, metrics, parameters, and model artifacts are tracked in MLflow
- The selected “best” (champion) model is persisted to a dedicated S3 location
- Inference services load models directly from S3.

---

### Serving & Inference Layer

The serving stack runs on a single **Amazon EC2** instance using Docker Compose.
- **FastAPI** exposes the prediction endpoints
- At application startup, the FastAPI service loads the latest champion model from S3 into memory
- **Streamlit** provides a lightweight frontend for interaction and demonstration
- Both services run in isolated **Docker** containers with registered images from **Amazon ECR**.
- The FastAPI service exposes a metrics endpoint that is scraped by **Prometheus** to perform monitoring and alerting.
- **Grafana** is used for the better visualization of metrics.

---

### CI/CD & Deployment Strategy

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

### Hyperparameter Tuning

The training pipeline uses optimized hyperparameters by the bayesian search of **Optuna**. MLflow logs a run at each trial of hyperparameter combination targeting the improvement of a metric value.
---

---

## ML Lifecycle Design





---

## Serving & Inference Strategy




---

## Project Structure



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
