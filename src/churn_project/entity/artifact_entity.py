from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    training_file_path: Path
    testing_file_path: Path
    feature_store_file_path: Path


@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: Path


@dataclass
class DataTransformationArtifact:
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_path: Path
    feature_names: list


@dataclass
class ModelTrainerArtifact:
    trained_model_path: Path
    model_registry_version: int
    metric_score: float
    model_name: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    model_evaluation_report_path: Path
