# End-to-End MLOps System for Churn Prediction


## Introduction

This repository contains a production-oriented end-to-end MLOps system for churn prediction in a bank. The project demonstrates core MLOps principles, including orchestration, model deployment, model versioning, experiment tracking, monitoring, automated retraining triggered by data drift, best practices and much more. 

**The core data science documentation with all the methodology context including exploratory data analysis, modeling and hyperparameter tuning is provided in [`research/Experiment.ipynb`](./research/Experiment.ipynb).**

---
## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [ML Lifecycle Design](#ml-lifecycle-design)
- [Setup](#setup)
- [Sample Run](#sample-run)
- [Limitations & Future Improvements](#limitations--future-improvements)

---
## Project Overview

### Problem Statement

Customer retention plays a critical role in maintaining long-term profitability in the modern banking environments. By predicting which customers are likely to leave in advance, companies can activate retention strategies. This project builds a predictive model to flag potential churners for a bank’s credit card services, enabling the bank to take targeted retention actions.

### High-level solution


**Check the project structure description in [`docs/project_structure.md`](./docs/project_structure.md)**

---
## System Architecture
![System Architecture](./docs/assets/Churn_Project_Architecture.svg)



#### 1. Orchestration

**Prefect** is a code-based orchestration tool that acts as the backbone of the training and monitoring workflows.
- The **training pipeline** orchestrates data ingestion, data validation, data transformation, model training, model evaluation and model pushing.
- The **data monitoring pipeline** runs independently. It consumes the **input** data, used for prediction and stored in an S3 Bucket, to perform a statistical test called Kolmogorov-Smirnov to detect data drift. Comparing the new coming data with a reference dataset, the monitoring pipeline triggers the training pipeline when drift thresholds are exceeded. Alongside, it generates and stores a data monitoring report using Evidently for diagnostics and visualization.

Prefect enables scheduled monitoring with deployed flows, which are registered and runnable versions of the pipelines.

#### 2. Experiment Tracking and Model Versioning

**MLflow** is used for experiment tracking and model versioning.
- All experiments, metrics, parameters, and model artifacts are tracked in MLflow
- The selected “best” (champion) model is persisted to a dedicated S3 location
- Inference services load models directly from S3.

#### 3. Hyperparameter Tuning

The training pipeline is feeded with optimized hyperparameters reached in the bayesian search of **Optuna** (check in the [experiment](./research/Experiment.ipynb)). MLflow logs a child run at each trial of hyperparameter combination targeting the improvement of a metric value. The best combination is provided in the parent run when the study is finished. The best combination of hyperparameters is passed to the training pipeline configuration in the params.yaml file.

#### 4. Serving & Inference Layer

The serving stack runs on a single **Amazon EC2** instance using Docker Compose.
- **FastAPI** exposes the prediction endpoints
- At application startup, the FastAPI service loads the latest champion model from S3 into memory
- **Streamlit** performs requests, providing a lightweight frontend for interaction and demonstration
- Both services run in isolated **Docker** containers with registered images from **Amazon ECR**.
- The FastAPI service exposes a metrics endpoint that is scraped by **Prometheus** to perform monitoring and alerting.
- **Grafana** is used for the better visualization of metrics.

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

**Ingestion → Validation → Transformation → Training → Evaluation → Promotion**

Each stage produces a well-defined artifact that is consumed by downstream stages, ensuring traceability, reproducibility, and failure isolation.

Each ML lifecycle stage was implemented in the following order:
- [config](./config)
- [entities](./src/churn_project/entity)
- [configuration manager](./src/churn_project/config/configuration.py)
- [components](./src/churn_project/components)
- [pipeline](./src/churn_project/orchestrator/training_flow.py)

**1. Data Ingestion:** Extract raw data from the source system and produce training and testing datasets.
* Data is extracted directly from a relational database using SQLAlchemy.
* A deterministic train/test split is applied using a fixed random seed.

**2. Data Validation:** Enforce schema integrity and data sanity before allowing any training to proceed.
* Column presence validation against the declared schema
* Data type validation per column
* Missing value detection

**3. Data Transformation:** Convert validated raw data into model-ready numerical representations.
* Feature engineering using a custom transformer
* Feature scaling using `StandardScaler`
* Class imbalance handling using **SMOTE**, applied **only to training data**

**4. Model Training**: Train a candidate model using validated, transformed data and register it.

Model Strategy:
* The model type is selected via configuration (e.g., XGBoost or Random Forest).
* Hyperparameters are defined externally and logged in config's params.yaml.
* Training occurs only on resampled, transformed data.

MLflow Integration:
* Training parameters and metrics are logged
* A complete inference pipeline (preprocessor + model) is registered in the MLflow Model Registry

Training only produces a **candidate**.

**5. Model Evaluation & Acceptance**: Decide whether a newly trained model is eligible for production.

Evaluation Logic:
* The current production (champion) model is loaded if it exists
* Both models are evaluated on the same untouched test dataset
* ROC-AUC is used as the primary selection metric

Acceptance Rule:
* A model is accepted **only if** it exceeds the production model’s AUC by a defined margin
* If no production model exists, the new model is accepted by default
* Each model version is explicitly tagged as `approved` or `rejected`

**6. Model Pushing:** Promote an approved model and make it available for inference.

Promotion Steps:
1. Approved models are copied to a dedicated production registry
2. The promoted version is assigned the alias `champion`
3. Model artifacts and metadata are exported and uploaded to a production S3 location

---

## Setup

This project supports two execution modes:

1. **Local Execution**  
   Used for experimentation, training, orchestration, and monitoring on a single machine.

2. **Production Deployment**  
   End-to-end deployment using Amazon ECR, EC2, and GitHub Actions.

This README's **Setup** is focused on **local execution**.  
Production deployment is documented in [`docs/deployment.md`](./docs/deployment.md).


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

**1.3. Install project dependencies**
```bash
poetry install
```
**1.4. Activate the virtual environment**
```bash
poetry env use python3.12
poetry env activate
eval "$(poetry env activate)"
```
If it wasn't activated, **restart your terminal**.

**1.5. Set up environment variables**

Create your local environment file:
```bash
cp .env.example .env
```
Edit the .env file and provide the necessary values ​​when you obtain them.

**1.6. Verify the setup**

You can perform a quick sanity check:
```bash
python -c "import churn_project; print('Environment OK')"
```

If no errors occur, the environment is correctly configured. **If you faced issues, check [`setup_issues.md`](./docs/setup_issues.md)**


### 2. Database Setup

**2.1. Download the dataset from Kaggle:**
[Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

**2.2. Add the dataset to MySQL**

- Remove non-informative (“naive”) columns
- Insert the dataset into a table named `churners` 

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

**3.2 You can access MLflow UI: [`http://127.0.0.1:5000`](http://127.0.0.1:5000)**

### 4. Prefect Setup (Orchestration)

- **Multiple terminals are recommended**:
    - Terminal 1: MLflow server
    - Terminal 2: Prefect server
    - Terminal 3: Prefect worker
    - Terminal 4 (optional): Training / inference commands

**4.1 Start the Prefect server running:**

```bash
prefect server start
```

**4.2 You can access Prefect UI: [`http://127.0.0.1:4200`](http://127.0.0.1:4200)**

**4.3 Set the Prefect API URL:**
```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

**4.3 Create a work pool**
```bash
prefect work-pool create churn-pool --type process
```

**4.4 Deploy flows**

**Training pipeline**

```bash
prefect deploy src/churn_project/orchestrator/training_flow.py:training_flow \
  -n churn-train
```
Configuration (for simplicity):
- Select the work-pool you just created (churn-pool)
- Remote code storage: No (recommended)
- Scheduling Configuration: No (This can be done in the UI)
- Save configuration: Yes (recommende)

**Data monitoring pipeline**

```bash
prefect deploy monitoring/data_drift/monitoring_flow.py:data_monitoring_flow \
  -n monitoring
```
You can use the same configuration of the previous prefect deployment

**4.5 Start a worker**

```bash
prefect worker start --pool churn-pool

```

### 5. S3 Bucket Setup

**5.1. Create an S3 bucket named `churn-production`**

**5.2 Create an IAM User with the following policy: `AmazonS3FullAccess`**

**The policy hereby used is useful just to reduce IAM complexity. In a real-world case, we would use a custom least-privilege policy.**

**5.3. Create an access key for the user and add the AWS credentials to .env**

---

**Now you're all set to run the project locally.**

**If you want to follow deployment, proceed with the complete setup in [`docs/deployment.md`](./docs/deployment.md).**

---

## Sample Run

### Hyperparameter Optimization
The Hyperparameter Optimization using Optuna can be performed running the experiment in [`research/Experiment.ipynb`](./research/Experiment.ipynb). The notebook is designed to allow for the optimization in two models: XGBoost and Random Forest. Performing both tuning, we get the following result in the MLflow UI:

![MLflow_Hyperparameter_Tuning](./docs/assets/MLflow_Hyperparameter_Tuning.png)

We can see the two parent runs corresponding to the runs of Optuna's study for the two models. Each parent run has its child runs corresponding to all the trials to reach the best combination of Hyperparameters. The resulting best combination for the corresponding model is provided clicking in its parent run. From there, we can pass these values to [`config/params.yaml`](./config/params.yaml) to feed training.

### Training Pipeline 
Run it with:
```bash
python main.py

```
or
```bash
prefect deployment run "TrainingPipelineFlow/churn-train"

```
The [`config/config.yaml`](./config/params.yaml) file has two very important values that can be modified to change the training approach:
- In the "model_trainer" configuration, we can set the model to be trained between "XGBClassifier" and "RandomForestClassifier"
- In the "model_evaluation" configuration, we can define the change threshold value, which indicates how much the AUC metric of the new trained model must be greater than the AUC of the old model in production for the new trained model to be accepted as the new production model → `is_new_trained_model_accepted = new_auc > (old_auc + threshold)`

It was performed 3 consecutive runs of the training pipeline in the following pattern:
1. `(model: RandomForestClassifier, threshold: 0.005)`
2. `(model: XGBClassifier, threshold: 0.005)`
3. `(model: XGBClassifier, threshold: 0.000)`

The **MLflow model registry** result follows:

![MLflow_Model_Registry](./docs/assets/MLflow_Model_Registry.png)

- On the left of the image is presented the "churn_model" registry, where all trained models are registered. The "validation_status" tag indicates whether the model outperformed the model currently in production.
- On the right of the image is presented the "prod.churn_model" registry, where all the approved models are registered. When a newly trained model outperforms the previous `champion` production model, it receives the alias `champion` and the old model loses the alias. The `champion` model will serve predictions.

The following image shows the **MLflow experiment tracking** result for the last run:

![MLflow_Training_Run](./docs/assets/MLflow_Training_Run.png)

### Services

To start the containers locally:
```bash
docker compose up -d
```
In the EC2 instance, containers will be started by CI/CD.

To access services:
| Service | Local | EC2 Instance |
| --- | --- | --- |
| FastAPI | [**`http://localhost:8000/health`**](http://localhost:8000/health) | **`http://<EC2_PUBLIC_IP>:8000/health`** |
| Streamlit | [**`http://localhost:8501`**](http://localhost:8501) | **`http://<EC2_PUBLIC_IP>:8501`** |
| Prometheus | [**`http://localhost:9090`**](http://localhost:9090) | **`http://<EC2_PUBLIC_IP>:9090`** |
| Grafana | [**`http://localhost:3000`**](http://localhost:3000) | **`http://<EC2_PUBLIC_IP>:3000`** |

#### Streamlit

![Streamlit](./docs/assets/Streamlit.png)

Streamlit performs requests for the both *post* endpoints of **FastAPI**: single prediction and batch prediction.

#### Grafana

![Grafana](./docs/assets/Grafana.png)

To reach the Grafana dashboard:
- We need to log into Grafana with user: admin and password: admin and then create a new password
- In "Connections", we add **Prometheus** as our data source
- We go to **Dashboards → Import**
- Paste the dashboard JSON file from [`monitoring/grafana/dashboard.json`](./monitoring/grafana/dashboard.json)
- When Grafana asks for a datasource mapping, set:\
   **DS_PROMETHEUS → your Prometheus datasource**
- Click **Import**. The dashboard loads immediately and shows our informative metrics.

### Data Monitoring Pipeline

The data monitoring pipeline uses the Kolmogorov-Smirnov statistical test to detect data drift in the input data used for prediction. When the test's p-value exceeds the threshold it means that the drift is statistically significant and retraining will be triggered. The default threshold is set to 0.05.

After some predictions in the streamlit interface, we can move on testing the data drift monitoring.

We can set scheduled monitoring in **[`Prefect UI`](http://127.0.0.1:4200) → Deployments → monitoring → Schedule** or run it manually with:  
```bash
prefect deployment run "DataMonitoringFlow/monitoring" \
  --params '{
    "date": "YYYY-MM-DD",
    "threshold": 0.05
  }'

```
*`replace YYYY-MM-DD with the date on which the predictions you want to monitor were performed (e.g. 2026-01-23)`*

To simplify the monitoring approach on this project, the monitoring takes into account only the data of the date (**in UTC time**) on which the predictions you want to monitor were performed. If monitoring is scheduled, the pipeline performs drift detection in the data from the scheduled date. If monitoring is triggered manually, the pipeline performs drift detection in the data from the specified date.

For the sample run, the Training Pipeline was triggered after detection of data drift by the Data Monitoring Pipeline for a certain date. The following image is provided by the Prefect UI, it shows the registered temporal sequence of tasks performed in the aforementioned scenario.

![Prefect_Monitoring](./docs/assets/Prefect_Monitoring.png)

The resulting evidently report follows:

![Evidently_Report](./docs/assets/Evidently_Report.png)

### CI/CD

![cicd](./docs/assets/cicd.png)

Containers up in the **EC2 Instance** and resulting **S3 Bucket**:

![S3_and_EC2](./docs/assets/S3_and_EC2.png)

---

## Limitations & Future Improvements

**This section highlights *the main* limitations of the current system to clarify design trade-offs and guide future improvements.**

### Data, Monitoring, and Retraining Limitations (Most Critical)
- **Retraining does not leverage drifted data distributions:**
When data drift is detected, retraining currently reuses the original training and test datasets sourced from MySQL. As a result, the newly trained model is not exposed to the shifted data distribution that triggered retraining, limiting the effectiveness of automated adaptation.

- **Lack of explicit data lineage across the ML lifecycle:**
Monitoring, retraining, and models are not yet linked through explicit data lineage. While drift detection triggers retraining, there is no guaranteed traceability between prediction logs, the data used for retraining, and the resulting model version.

- **Reference data is static and not versioned per model:**
The monitoring pipeline relies on a single reference dataset. There is currently no mechanism to maintain reference data aligned with each production model version, which may reduce the accuracy of drift detection after multiple deployments.

- **Drift detection is based on single-day snapshots:**
Data drift detection is performed on a per-day snapshot of prediction data. This approach is sensitive to short-term fluctuations and does not capture longer-term distributional trends, increasing the risk of false-positive retraining triggers.

- **Overly sensitive retraining trigger logic:**
Retraining is triggered as soon as any single feature is detected as drifted, with no minimum number of drifted features or cooldown mechanism. This can lead to unnecessary retraining cycles caused by transient or low-impact distribution changes.

- **Prediction logs are stored as individual S3 objects:**
Each prediction request is logged as a separate JSON object in S3. While simple and transparent, this approach does not scale efficiently and complicates downstream aggregation, querying, and dataset construction for retraining.

### Model Deployment and Serving Limitations

- **Model updates require API restart or redeployment:**
The prediction API loads the production model at application startup. Newly promoted models are therefore not detected automatically, requiring a service restart or redeployment to serve updated models.

- **Model serving is tightly coupled to the application:**
The prediction service is embedded directly within the FastAPI application. This coupling limits flexibility, complicates independent scaling of inference workloads, and prevents adopting alternative serving strategies without application changes.

### Security and Operational Limitations

- **Prometheus metrics endpoint is publicly exposed:**
The /metrics endpoint is currently accessible without authentication. In a production environment, this could expose internal system behavior and should be protected via network-level controls or authentication mechanisms.
