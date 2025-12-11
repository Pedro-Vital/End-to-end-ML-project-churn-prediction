import json
import os
import sys
import tempfile
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from churn_project.aws.s3_utils import upload_folder_to_s3
from churn_project.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from churn_project.entity.config_entity import ModelPusherConfig
from churn_project.exception import CustomException
from churn_project.logger import logger


class ModelPusher:
    def __init__(self, config: ModelPusherConfig):
        self.config = config
        self.mlflow_config = config.mlflow_config
        self.client = MlflowClient(tracking_uri=self.config.mlflow_config.tracking_uri)
        # production S3 URI where API will fetch the model
        self.prod_s3_uri = self.config.prod_s3_uri

    def promote_in_mlflow(self, source_registry_version: int):
        """
        Core promotion logic:
        1. Tag model as approved in MLflow
        2. Copy to production environment
        3. Assign alias 'champion' to the promoted model version
        """
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

    def deploy_to_s3(self, prod_version: int):
        """
        Upload model artifacts and metadata to production S3 location
        """
        logger.info("Uploading model artifacts to production S3 location.")

        model_version = self.client.get_model_version(
            name=self.mlflow_config.prod_registry_name, version=prod_version
        )
        artifact_uri = model_version.source
        run_id = model_version.run_id

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model artifacts to temp directory
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=os.path.join(temp_dir, "model"),
            )
            logger.info(f"Downloaded model artifacts to {local_path}")

            # Write metadata to the same temp directory
            metadata = {
                "version": prod_version,
                "promoted_at": str(datetime.isoformat(datetime.now())),
                "run_id": run_id,
            }

            with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)
            logger.info(f"Wrote metadata to {f.name}")

            # Upload entire temp directory to S3
            upload_folder_to_s3(folder_path=temp_dir, s3_uri=self.prod_s3_uri)
            logger.info(f"Uploaded model and metadata to {self.prod_s3_uri}")

            # Extract model requirements file and save for CI / Docker build
            dst_path = os.path.join("artifacts", "infra", "model")
            os.makedirs(dst_path, exist_ok=True)
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"{artifact_uri}/requirements.txt",
                dst_path=dst_path,
            )

    def push_model(self, version: int):
        """
        Push model to production by promoting in MLflow and deploying to S3
        """
        logger.info("Pushing model to production.")
        prod_version = self.promote_in_mlflow(version)
        self.deploy_to_s3(prod_version=prod_version)
        logger.info("Model pushed to production successfully.")

    def initiate_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelPusherArtifact:
        try:
            logger.info("Starting model pusher process.")

            if model_evaluation_artifact.is_model_accepted:
                self.push_model(version=model_trainer_artifact.registry_version)
                promoted = {"promoted": True}
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

            return ModelPusherArtifact(promoted=promoted)
        except Exception as e:
            logger.error(f"Error in model pusher process: {e}")
            raise CustomException(e, sys)
