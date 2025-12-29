# Setup

This project supports two execution modes:

1. **Local Execution**  
   Used for experimentation, training, orchestration, and monitoring on a single machine.

2. **Production Deployment**  
   End-to-end deployment using Amazon ECR, EC2, and GitHub Actions.

This **Setup** is focused on **local execution**.  
Production deployment is documented in [`deployment.md`](./deployment.md).


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

If no errors occur, the environment is correctly configured. **If you faced issues, check [`setup_issues.md`](./setup_issues.md)**


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

***The next steps of prefect setup are only required for the Data Monitoring Pipeline usage, you can bypass it if you want just training and services.***

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

**Now you're all set to run the project locally. You can check the sample run in [`README.md`](../README.md) to know how to run it**

**If you want to follow deployment, proceed with the complete setup in [`deployment.md`](./deployment.md).**
