import sys
from datetime import datetime

import pandas as pd

from churn_project.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from churn_project.entity.config_entity import DataValidationConfig
from churn_project.exception import CustomException
from churn_project.logger import logger
from churn_project.utils import save_json


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_columns(self, data: pd.DataFrame) -> bool:
        """Checks whether all the columns exist in the dataframe"""
        try:
            missing_columns = []
            for column in self.config.columns:
                if column not in data.columns:
                    missing_columns.append(column)

            if missing_columns:
                logger.info(f"Missing columns: {missing_columns}")
                return False
            logger.info("All columns are present.")
            return True
        except Exception as e:
            logger.error(f"Error occurred while checking for column existence: {e}")
            raise CustomException(e, sys)

    def validate_data_types(self, data: pd.DataFrame) -> bool:
        """Checks whether the data types of columns match the expected types"""
        try:
            mismatched_types = {}
            for column, expected_type in self.config.columns.items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if actual_type != expected_type:
                        mismatched_types[column] = {
                            "expected": expected_type,
                            "actual": actual_type,
                        }

            if mismatched_types:
                logger.info(f"Mismatched data types: {mismatched_types}")
                return False
            logger.info("All data types match the expected types.")
            return True
        except Exception as e:
            logger.error(f"Error occurred while validating data types: {e}")
            raise CustomException(e, sys)

    def check_missing_values(self, data: pd.DataFrame) -> bool:
        """Checks for missing values in the dataframe"""
        try:
            missing_value_report = data.isnull().sum().to_dict()
            total_missing = sum(missing_value_report.values())
            if total_missing > 0:
                logger.info(f"Missing values found: {missing_value_report}")
                return False
            logger.info("No missing values found.")
            return True
        except Exception as e:
            logger.error(f"Error occurred while checking for missing values: {e}")
            raise CustomException(e, sys)

    def initiate_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Main method to initiate data validation process"""
        try:
            logger.info("Starting data validation process")

            # Load the ingested data
            data = pd.read_csv(data_ingestion_artifact.raw_data_path)

            # Validate columns
            is_column_valid = self.validate_columns(data)
            is_type_valid = self.validate_data_types(data)
            is_missing_values_valid = self.check_missing_values(data)

            validation_status = (
                is_column_valid and is_type_valid and is_missing_values_valid
            )

            validation_report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": validation_status,
                "check": {
                    "columns": "Success" if is_column_valid else "Failed",
                    "data_types": "Success" if is_type_valid else "Failed",
                    "missing_values": (
                        "Success" if is_missing_values_valid else "Failed"
                    ),
                },
                "sample": data.head().to_dict(),
            }
            save_json(self.config.validation_report_path, validation_report)
            logger.info(
                f"Data validation report saved at {self.config.validation_report_path}"
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                validation_report_path=self.config.validation_report_path,
            )

            logger.info("Data validation process completed")
            return data_validation_artifact
        except Exception as e:
            logger.error(f"Error occurred during data validation: {e}")
            raise CustomException(e, sys)
