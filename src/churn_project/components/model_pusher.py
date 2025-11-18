import sys

import mlflow

from churn_project.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from churn_project.entity.config_entity import ModelPusherConfig
from churn_project.exception import CustomException
from churn_project.logger import logger


class ModelPusher:
    def __init__(self, config: ModelPusherConfig):
        self.config = config
        self.mlflow_config = config.mlflow_config
        self.client = mlflow.tracking.MlflowClient(
            tracking_uri=self.config.mlflow_config.tracking_uri
        )

    def promote_model(self, source_registry_version: int):
        logger.info("Promoting model to production stage.")
        # 1. Tag model as approved in MLflow
        self.client.set_model_version_tag(
            name=self.mlflow_config.registry_name,
            version=source_registry_version,
            key="validation_status",
            value="approved",
        )

        # 2. Copy to production environment
        new_model_uri = (
            f"models:/{self.mlflow_config.registry_name}/{source_registry_version}"
        )
        dst_name = self.mlflow_config.prod_registry_name
        logger.info(
            f"Copying model version from {new_model_uri} to registry {dst_name}"
        )
        dst_version = self.client.copy_model_version(
            src_model_uri=new_model_uri,
            dst_name=dst_name,
        )
        logger.info(f"Copied version to {dst_name} as {dst_version.version}")

        # 3. Assign alias 'champion' to the promoted model version
        self.client.set_registered_model_alias(
            name=dst_name,
            alias="champion",
            version=dst_version.version,
        )
        return dst_version.version

    def initiate_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> dict:
        try:
            logger.info("Starting model pusher process.")

            if model_evaluation_artifact.is_model_accepted:
                # Promote the model to production
                prod_version = self.promote_model(
                    model_trainer_artifact.registry_version
                )
                promoted = {"promoted": True, "prod_version": prod_version}
            else:
                self.client.set_model_version_tag(
                    name=self.mlflow_config.registry_name,
                    version=model_trainer_artifact.registry_version,
                    key="validation_status",
                    value="rejected",
                )
                logger.info("Model not accepted. Skipping promotion.")
                promoted = {"promoted": False}

            logger.info("Model pusher process completed successfully.")

            return promoted
        except Exception as e:
            logger.error(f"Error in model pusher process: {e}")
            raise CustomException(e, sys)
