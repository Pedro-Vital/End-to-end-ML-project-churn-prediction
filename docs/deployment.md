# Deployment Setup

**To follow this guide, it is assumed that the steps in [`setup.md`](./setup.md) have been completed and that you have subsequently run the training pipeline at least once: `python main.py` to run it.**

## Deployment Architecture

- GitHub Actions builds Docker images
- Images are pushed to Amazon ECR
- EC2 pulls images and runs Docker Compose
- The EC2 instance does not build images
- Deployment is triggered by pushing changes to the repository

### 1. Containers Setup

**1.1 Create two Private ECR Repositories: `churn-fastapi` and `churn-streamlit`**\
Configuration:
- Image tag mutability: Mutable
- Enable Scan on push
- Default for the rest

**1.2 Replace the AWS ACCOUNT ID and AWS REGION of [`docker-compose.prod.yaml`](../docker-compose.prod.yaml) with YOUR actual AWS ACCOUNT ID and AWS REGION to match your image URIs, but keep the "latest" tag.**

### 2. IAM Setup

**2.1 Create an IAM User with the following policy: `AmazonEC2ContainerRegistryPowerUser`**\
It will be used by GitHub Actions

**2.1 Create an IAM Role named `churn-ec2-role` with the following policies: `AmazonEC2ContainerRegistryReadOnly` and `AmazonS3FullAccess`**\
Configuration:
- Trusted entity type: AWS Service
- Use case: EC2

**The policy hereby used is useful just to reduce IAM complexity. In a real-world case, we would use a custom least-privilege policy.**

### 3. EC2 Bootstrap

**3.1 Launch EC2**
- Name: churn-prod-ec2
- OS: Amazon Linux 2023 (AL2023)
- Instance type: t3.small
- Create or Select Key Pair (RSA). You will later store this key as a GitHub Secret or you can use it to SSH into the EC2 instance.
- VPC: Default
- Subnet: Default
- Auto-assign Public IP: Enable
- Assign it a security group named churn-prod-sg with the following inbound rules:

| Port | Source | Purpose |
| --- | --- | --- |
| 22 | 0.0.0.0/0 | SSH |
| 8000 | 0.0.0.0/0 | FastAPI |
| 8501 | 0.0.0.0/0 | Streamlit |
| 9090 | 0.0.0.0/0 | Prometheus |
| 3000 | 0.0.0.0/0 | Grafana |

**Warning: These inbound rules are for quick start, they are not recommended long-term. The most appropriate is to set the source for SSH, Prometheus and Grafana as your IP and connect to EC2 via SSH Client**

- Configure storage: 30 GB, type gp3
- IAM Instance Profile (Advanced Details): churn-ec2-role
- Default / Empty for the rest

**Launch Instance**

**3.2 Install Docker + Compose**
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

***The GitHub Actions workflow will automatically trigger deployment by pushing changes to the repository. To follow that approach, you just need to set the GitHub Secrets and add a minimal modification to the repository.*** 

### 4. GitHub Secrets Setup

**4.1 Add the following as variables**
- AWS_ACCOUNT_ID → Your AWS account ID
- AWS_REGION → Your AWS Region
- EC2_USER → if you didn't change it, it is `ec2-user` (Amazon Linux)

**4.2 Add the following as secrets**
- AWS_ACCESS_KEY_ID → IAM user access key for ECR auth only (from 7.1)
- AWS_SECRET_ACCESS_KEY → IAM user secret key for ECR auth only (from 7.1)
- EC2_HOST → Your Public IPv4 or DNS
- EC2_SSH_KEY → Private key → That is the key selected at EC2 launch. Use `cat <key-name>` to access, then copy and paste it entirely to secrets.

### To confirm services are reachable:

- FastAPI: **`http://<EC2_PUBLIC_IP>:8000/health`**
- Streamlit: **`http://<EC2_PUBLIC_IP>:8501`**
- Prometheus: **`http://<EC2_PUBLIC_IP>:9090`**
- Grafana: **`http://<EC2_PUBLIC_IP>:3000`**

***But if you want to deploy manually without using the GitHub Actions workflow, follow the instructions below.***
---
---
# Manual deploy

Manual deployment assumes Docker images are already built and pushed to the container registry. If this is the first deployment, or if images were never pushed before, you must build and push them manually before deploying on EC2.

**Prerequisites (Local Machine)**

Ensure you have:
- Docker installed and running
- AWS CLI configured locally
- Permissions to push to Amazon ECR
  - `aws login`
  - `aws ecr get-login-password --region <AWS_REGION>`
    - If this command fails with `AccessDeniedException`, you do not have sufficient permissions.

### 1. Manual Image Build & Push (from local machine)

**1.1. Authenticate Docker to ECR:**
```bash
aws ecr get-login-password --region <AWS_REGION> \
  | docker login \
    --username AWS \
    --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com
```

**1.2 Build and Push FastAPI Image (from churn-project directory)**
```bash
docker build --pull \
  -f src/churn_project/api/Dockerfile \
  -t churn-fastapi:latest .

docker tag churn-fastapi:latest \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/churn-fastapi:latest

docker push \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/churn-fastapi:latest

```

**1.2 Build and Push Streamlit Image (from churn-project directory)**
```bash
docker build --pull \
  -f frontend/Dockerfile \
  -t churn-streamlit:latest .

docker tag churn-streamlit:latest \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/churn-streamlit:latest

docker push \
  <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/churn-streamlit:latest


```
Then proceed with manual deployment on EC2

### 2. Manual Deployment Guide

**2.1 Connect to the EC2 Instance**

You can connect via AWS Console or SSH Client:
```bash
ssh -i <~/path-to-your/churn-ec2-key.pem> ec2-user@<EC2_PUBLIC_IP>

```

**2.2 Prepare the Deployment Directory (from EC2 Instance)**

The project expects all runtime configuration to live under /opt/churn.
```bash
sudo mkdir -p /opt/churn/monitoring/prometheus
sudo chown -R ec2-user:ec2-user /opt/churn
```

Expected structure:
```
/opt/churn
├── docker-compose.prod.yaml
└── monitoring/
    └── prometheus/
        ├── prometheus.yml
        └── alert_rules.yml

```

**2.3. Copy Required Files to EC2 (from local machine)**
```bash
scp -i <~/path-to-your/churn-ec2-key.pem> docker-compose.prod.yaml \
  ec2-user@<EC2_PUBLIC_IP>:/opt/churn/

scp -i <~/path-to-your/churn-ec2-key.pem> -r monitoring/prometheus \
  ec2-user@<EC2_PUBLIC_IP>:/opt/churn/monitoring/
```

**2.4. Authenticate Docker with Amazon ECR (from EC2 Instance)**
```bash
aws ecr get-login-password --region <AWS_REGION> \
  | docker login \
    --username AWS \
    --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com

```
If the EC2 instance has an IAM role attached, no AWS keys are required.

**2.5. Pull the Latest Images from ECR (from EC2 Instance)**

Navigate to the deployment directory:
```bash
cd /opt/churn
```
Pull the latest images:
```bash
docker compose -f docker-compose.prod.yaml pull
```

**2.6. Start (or Update) the Services (from EC2 Instance)**
```bash
docker compose -f docker-compose.prod.yaml up -d --remove-orphans
```
This will start containers if they are not running, recreate containers if images changed and remove obsolete containers if exists.

**2.7 Confirm services are reachable:**

- FastAPI: **`http://<EC2_PUBLIC_IP>:8000/health`**
- Streamlit: **`http://<EC2_PUBLIC_IP>:8501`**
- Prometheus: **`http://<EC2_PUBLIC_IP>:9090`**
- Grafana: **`http://<EC2_PUBLIC_IP>:3000`**
