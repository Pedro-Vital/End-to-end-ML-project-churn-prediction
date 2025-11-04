import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from churn_project.entity.artifact_entity import DataValidationArtifact
from churn_project.entity.config_entity import DataValidationConfig
from churn_project.exception import CustomException
from churn_project.logger import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def is_columns_exist(self, data: pd.DataFrame) -> bool:
        """
        Method Name :   is_columns_exist
        Description :   This method checks whether all the columns exist in the dataframe

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            missing_columns = []
            for column in self.config.all_schema:
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

    def detect_data_drift(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> bool:
        """
        Method Name :   detect_data_drift
        Description :   This method detects data drift between reference and current datasets

        Output      :   Returns bool value - True if drift is detected, False otherwise
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logger.info("Starting data drift detection")

            # Identify numerical and categorical columns
            numerical_columns = reference_data.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()
            categorical_columns = reference_data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Define the data schema
            schema = DataDefinition(
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
            )

            # Create Evidently datasets with the defined schema
            eval_data_curr = Dataset.from_pandas(current_data, data_definition=schema)
            eval_data_ref = Dataset.from_pandas(reference_data, data_definition=schema)

            # Create and run the drift report
            report = Report([DataDriftPreset()])
            report = report.run(
                reference_data=eval_data_ref, current_data=eval_data_curr
            )

            # Save the report as HTML
            report.save_html(str(self.config.drift_report_file_path))
            logger.info(f"Drift report saved to {self.config.drift_report_file_path}")

            # Get the report results as dictionary
            report_dict = report.dict()

            # Check if dataset drift is detected

            drift_detected = False
            if report_dict.get("metrics")[0]['value']['count'] > 0:
                logger.info("Data drift detected!")
                drift_detected = True
            else:
                logger.info("No data drift detected.")

            return drift_detected

        except Exception as e:
            logger.error(f"Error occurred during data drift detection: {e}")
            raise e

    def initiate_data_validation(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation process")

            # Check if all columns exist in both datasets
            if not self.is_columns_exist(reference_data):
                raise CustomException(
                    Exception(
                        "Reference/ training data is missing some required columns."
                    ),
                    sys,
                )
            if not self.is_columns_exist(current_data):
                raise CustomException(
                    Exception(
                        "Current/ testing data is missing some required columns."
                    ),
                    sys,
                )

            # Detect data drift
            drift_detected = self.detect_data_drift(reference_data, current_data)

            # Create DataValidationArtifact
            validation_artifact = DataValidationArtifact(
                validation_status=not drift_detected,
                message="Data validation completed successfully.",
                drift_report_file_path=Path(self.config.drift_report_file_path),
            )

            logger.info("Data validation process completed")
            return validation_artifact

        except Exception as e:
            logger.error(f"Error occurred during data validation: {e}")
            raise CustomException(e, sys)
