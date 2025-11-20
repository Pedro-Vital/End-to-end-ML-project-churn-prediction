from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    training_path: Path
    testing_path: Path
    raw_data_path: Path


@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: Path


@dataclass
class DataTransformationArtifact:
    transformed_train_path: Path
    preprocessor_path: Path
    feature_names: list
    raw_train_path: Path
    raw_test_path: Path


@dataclass
class ModelTrainerArtifact:
    registry_version: int


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    model_evaluation_report_path: Path
