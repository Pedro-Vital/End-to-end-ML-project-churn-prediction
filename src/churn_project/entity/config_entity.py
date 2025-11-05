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
