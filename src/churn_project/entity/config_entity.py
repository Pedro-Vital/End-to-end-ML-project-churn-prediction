from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    db_host: str
    db_user: str
    db_password: str
    db_name: str
    base_query: str
    feature_store_file_path: Path
    training_file_path: Path
    testing_file_path: Path
    train_test_split_ratio: float
    random_state: int
    columns: dict


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    validation_report_path: Path
    columns: dict


@dataclass
class DataTransformationConfig:
    root_dir: Path
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_path: Path
    target_column: str
    drop_columns: list[str]
    random_state: int


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    model_name: str
    registry_name: str
    best_params: dict
    mlflow_uri: str
    mlflow_experiment_name: str


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_evaluation_report_path: Path
    registry_name: str
    prod_registry_name: str
    mlflow_uri: str
    mlflow_experiment_name: str


# @dataclass
# class ModelPusherConfig:
#     root_dir: Path
#     deployed_model_path: Path
#     registry_name: str
#     mlflow_uri: str
#     mlflow_experiment_name: str
