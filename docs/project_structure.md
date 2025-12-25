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
