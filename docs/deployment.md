# Deployment Setup

**To follow this guide, it is assumed that all steps in the Setup section of the README file have been completed.**

---
## Deployment Architecture

- GitHub Actions builds Docker images
- Images are pushed to Amazon ECR
- EC2 pulls images and runs Docker Compose
- The EC2 instance does not build images
- Deployment is triggered by pushing changes to the repository

### 1. Containers Setup

**1.1 Create two ECR Repositories: `churn-fastapi` and `churn-streamlit`**\
Configuration:
- Visibility: Private
- Image tag mutability: Mutable
- Enable Scan on push
- Default for the rest

**1.2 Click each repository and copy the repository URI**

**1.3 Replace the repository URI and AWS_REGION of [docker-compose.prod.yaml](./docker-compose.prod.yaml) with your actual repository URI and region**

### 2. IAM Setup

**2.1 Create an IAM User with the following policy: `AmazonEC2ContainerRegistryPowerUser`**\
It will be used by GitHub Actions

**2.1 Create an IAM Role named `churn-ec2-role` with the following policies: `AmazonEC2ContainerRegistryReadOnly` and `AmazonS3FullAccess`**\
Configuration:
- Use case: EC2

**The policy hereby used is useful just to reduce IAM complexity. In a real-world case, we would use a custom least-privilege policy.**

### 3. EC2 Bootstrap

**3.1 Launch EC2**
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

### 4 GitHub Secrets Setup

**4.1 Add the following as variables**
- AWS_ACCOUNT_ID → Your AWS account ID
- AWS_REGION → Your AWS Region
- EC2_USER → if you didn't change it, it is `ec2-user` (Amazon Linux)

**4.2 Add the following as secrets**
- AWS_ACCESS_KEY_ID → IAM user access key for ECR auth only (from 7.1)
- AWS_SECRET_ACCESS_KEY → IAM user secret key for ECR auth only (from 7.1)
- EC2_HOST → Your Public IPv4 or DNS
- EC2_SSH_KEY → Private key → That is the key selected at EC2 launch. Use `cat <key-name>` to access, then copy and paste it entirely to secrets.

---
***Deployments are triggered by pushing changes to the repository. No deployment commands are executed locally.***

***So now you can run deployment adding a minimal modification to the repository. It will trigger GitHub Actions workflow.***
