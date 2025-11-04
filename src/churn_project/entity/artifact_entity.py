from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    training_file_path: Path
    testing_file_path: Path


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: Path
