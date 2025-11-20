import sys
from typing import Tuple

import mlflow

from churn_project.components.data_ingestion import DataIngestion
from churn_project.components.data_transformation import DataTransformation
from churn_project.components.data_validation import DataValidation
from churn_project.components.model_evaluation import ModelEvaluation
from churn_project.components.model_pusher import ModelPusher
from churn_project.components.model_trainer import ModelTrainer
from churn_project.config.configuration import ConfigurationManager
from churn_project.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from churn_project.exception import CustomException
from churn_project.logger import logger


class TrainingPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.mlflow_config = config_manager.get_mlflow_config()
        self.data_ingestion_config = config_manager.get_data_ingestion_config()
        self.data_validation_config = config_manager.get_data_validation_config()
        self.data_transformation_config = (
            config_manager.get_data_transformation_config()
        )
        self.model_trainer_config = config_manager.get_model_trainer_config()
        self.model_evaluation_config = config_manager.get_model_evaluation_config()
        self.model_pusher_config = config_manager.get_model_pusher_config()

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

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        This method starts the data validation component of the training pipeline.
        """
        try:
            logger.info("Starting data validation component of the training pipeline.")

            data_validation = DataValidation(config=self.data_validation_config)

            # Initiate data validation
            data_validation_artifact = data_validation.initiate_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logger.info("Data validation component completed successfully.")
            return data_validation_artifact

        except Exception as e:
            logger.error(f"Error in data validation component: {e}")
            raise CustomException(e, sys)

    def start_data_transformation(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataTransformationArtifact:
        """
        This method starts the data transformation component of the training pipeline.
        """
        try:
            if not data_validation_artifact.validation_status:
                raise Exception(
                    "Data validation failed. Cannot proceed to data transformation."
                )

            logger.info(
                "Starting data transformation component of the training pipeline."
            )

            data_transformation = DataTransformation(
                config=self.data_transformation_config
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            )

            logger.info("Data transformation component completed successfully.")
            return data_transformation_artifact

        except Exception as e:
            logger.error(f"Error in data transformation component: {e}")
            raise CustomException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        This method starts the model training component of the training pipeline.
        """
        try:
            logger.info("Starting model training component of the training pipeline.")

            model_trainer = ModelTrainer(config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logger.info("Model training component completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            logger.error(f"Error in model training component: {e}")
            raise CustomException(e, sys)

    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        """
        This method starts the model evaluation component of the training pipeline.
        """
        try:
            logger.info("Starting model evaluation component of the training pipeline.")

            model_evaluation = ModelEvaluation(config=self.model_evaluation_config)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            logger.info("Model evaluation component completed successfully.")
            return model_evaluation_artifact

        except Exception as e:
            logger.error(f"Error in model evaluation component: {e}")
            raise CustomException(e, sys)

    def start_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelPusherArtifact:
        """
        This method starts the model pusher component of the training pipeline.
        """
        try:
            logger.info("Starting model pusher component of the training pipeline.")

            model_pusher = ModelPusher(config=self.model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            logger.info("Model pusher component completed successfully.")
            return model_pusher_artifact

        except Exception as e:
            logger.error(f"Error in model pusher component: {e}")
            raise CustomException(e, sys)

    def run_pipeline(
        self,
    ) -> Tuple[
        DataIngestionArtifact,
        DataValidationArtifact,
        DataTransformationArtifact,
        ModelTrainerArtifact,
        ModelEvaluationArtifact,
        ModelPusherArtifact,
    ]:
        """Run the entire training pipeline"""
        try:
            logger.info("Starting training pipeline")

            # Set up MLflow tracking URI and experiment
            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            with mlflow.start_run(run_name="Pipeline_Run"):
                mlflow.set_tag("developer", "Pedro")

                data_ingestion_artifact = self.start_data_ingestion()
                data_validation_artifact = self.start_data_validation(
                    data_ingestion_artifact
                )
                data_transformation_artifact = self.start_data_transformation(
                    data_validation_artifact, data_ingestion_artifact
                )
                model_trainer_artifact = self.start_model_trainer(
                    data_transformation_artifact
                )
                model_evaluation_artifact = self.start_model_evaluation(
                    data_transformation_artifact, model_trainer_artifact
                )
                model_pusher_artifact = self.start_model_pusher(
                    model_evaluation_artifact, model_trainer_artifact
                )

            logger.info("Training pipeline completed successfully")
            return (
                data_ingestion_artifact,
                data_validation_artifact,
                data_transformation_artifact,
                model_trainer_artifact,
                model_evaluation_artifact,
                model_pusher_artifact,
            )

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)
