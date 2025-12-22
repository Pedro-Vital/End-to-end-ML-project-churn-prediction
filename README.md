# End-to-End MLOps System for Churn Prediction

---

## Introduction

This repository contains an end-to-end MLOps system for churn prediction in a bank. The project demonstrates core MLOps principles, including orchestration, model deployment, model versioning, experiment tracking, monitoring, automated retraining triggered by data drift, best practices and much more. The core data science methodology including EDA, modeling and the necessary hyperparameter tuning using bayesian search to feed the training pipeline is provided in the [Experiment.ipynb](./research/Experiment.ipynb).

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Data](#data)
- [Experiment Tracking](#experiment-tracking)
- [Orchestration](#orchestration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)
- [Setup](#setup)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Future Works](#future-works)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement



---

## System Architecture
![System Architecture](.docs/assets/Churn_Project_Architecture.svg)



---

## Data





---

## Experiment Tracking and Model Versioning



---

## Orchestration




---

## Deployment



---

## Best Practices




---

## Setup




---

## Project Structure



---

## Sample Run






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


#### Drift Detection Strategy

This system uses a two-layer monitoring approach:

* Statistical layer (SciPy KS test) is the authoritative signal used for automation and retraining decisions.

* Observability layer (Evidently) is used exclusively for diagnostics, visualization, and human inspection.

Evidently results do not trigger automated actions.
