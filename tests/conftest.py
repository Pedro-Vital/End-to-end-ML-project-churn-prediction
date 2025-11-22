import pandas as pd
import pytest

from churn_project.entity.config_entity import MlflowConfig, ModelPredictorConfig
from churn_project.predictor.predict import ModelPredictor


@pytest.fixture
def mlflow_config():
    """Fixture for MLflow configuration."""
    return MlflowConfig(
        tracking_uri="http://127.0.0.1:5000",
        prod_registry_name="prod.churn_model",
        registry_name="churn_model",
        experiment_name="churn_prediction",
    )


@pytest.fixture
def predictor_config(mlflow_config):
    """Fixture for ModelPredictor config."""
    return ModelPredictorConfig(mlflow_config=mlflow_config)


@pytest.fixture
def predictor(predictor_config):
    """Fixture that returns a ModelPredictor instance."""
    return ModelPredictor(predictor_config)


@pytest.fixture
def sample_input():
    """Sample input with valid schema."""
    return pd.DataFrame(
        {
            "Total_Relationship_Count": [5.0],
            "Credit_Limit": [12691.0],
            "Total_Revolving_Bal": [2517.0],
            "Total_Amt_Chng_Q4_Q1": [1335.0],
            "Total_Trans_Amt": [1350.0],
            "Total_Trans_Ct": [24.0],
            "Total_Ct_Chng_Q4_Q1": [1.2],
            "Avg_Utilization_Ratio": [0.198],
        }
    )
