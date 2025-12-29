# ML Lifecycle Design

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
* Hyperparameters are defined using the [experiment](./research/Experiment.ipynb) and logged in [`config/params.yaml`](./config/params.yaml).
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
