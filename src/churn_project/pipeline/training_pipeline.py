import sys

import pandas as pd

from churn_project.components.data_ingestion import DataIngestion
from churn_project.components.data_validation import DataValidation
from churn_project.config.configuration import ConfigurationManager
from churn_project.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from churn_project.exception import CustomException
from churn_project.logger import logger


class TrainingPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.data_ingestion_config = config_manager.get_data_ingestion_config()
        self.data_validation_config = config_manager.get_data_validation_config()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method starts the data ingestion component of the training pipeline.
        """
        try:
            logger.info("Starting data ingestion component of the training pipeline.")

            data_ingestion = DataIngestion(config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logger.info("Data ingestion component completed successfully.")

            return data_ingestion_artifact
        except Exception as e:
            logger.error(f"Error in data ingestion component: {e}")
            raise CustomException(e, sys)

    def start_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation component of the training pipeline.")

            # Load reference and current datasets
            data_ref = pd.read_csv(self.data_ingestion_artifact.training_file_path)
            data_curr = pd.read_csv(self.data_ingestion_artifact.testing_file_path)

            data_validation = DataValidation(config=self.data_validation_config)

            # Initiate data validation
            data_validation_artifact = data_validation.initiate_data_validation(
                reference_data=data_ref, current_data=data_curr
            )

            logger.info("Data validation component completed successfully.")

            return data_validation_artifact
        except Exception as e:
            logger.error(f"Error in data validation component: {e}")
            raise CustomException(e, sys)

    def run_pipeline(self) -> None:
        """
        This method runs the entire training pipeline.
        """
        try:
            logger.info("Starting the training pipeline.")

            # Start data ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            self.data_ingestion_artifact = data_ingestion_artifact

            # Start data validation
            data_validation_artifact = self.start_data_validation()
            self.data_validation_artifact = data_validation_artifact

            logger.info("Training pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)
