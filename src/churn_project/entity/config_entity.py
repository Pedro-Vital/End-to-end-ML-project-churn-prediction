from dataclasses import dataclass
from pathlib import Path


@dataclass
class MlflowConfig:
    tracking_uri: str
    experiment_name: str
    registry_name: str
    prod_registry_name: str


@dataclass(frozen=True)
class DataIngestionConfig:
    db_host: str
    db_user: str
    db_password: str
    db_name: str
    base_query: str
    raw_data_path: Path
    training_path: Path
    testing_path: Path
    train_test_split_ratio: float
    random_state: int
    columns: dict


@dataclass(frozen=True)
class DataValidationConfig:
    validation_report_path: Path
    columns: dict


@dataclass
class DataTransformationConfig:
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_path: Path
    target_column: str
    drop_columns: list[str]
    random_state: int


@dataclass
class ModelTrainerConfig:
    model_name: str
    target_column: str
    best_params: dict
    mlflow_config: MlflowConfig


@dataclass
class ModelEvaluationConfig:
    target_column: str
    model_evaluation_report_path: Path
    change_threshold: float
    mlflow_config: MlflowConfig


@dataclass
class ModelPusherConfig:
    mlflow_config: MlflowConfig


@dataclass
class ModelPredictorConfig:
    mlflow_config: MlflowConfig
