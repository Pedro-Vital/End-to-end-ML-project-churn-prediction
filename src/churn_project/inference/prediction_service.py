import sys
from datetime import datetime

import mlflow
import pandas as pd

from churn_project.entity.config_entity import MlflowConfig
from churn_project.exception import CustomException
from churn_project.logger import logger


class PredictionService:
    """
    Production-ready Prediction Service that loads the champion inference pipeline
    from the MLflow Model Registry and performs predictions on new data.

    Designed to be instantiated once (singleton) inside FastAPI applications.
    """

    def __init__(self, mlflow_config: MlflowConfig):
        """
        Initialize and store configuration.

        Args:
            mlflow_config: MlflowConfig containing MLflow configuration
        """
        self.mlflow_config = mlflow_config
        self.model, self.model_version = self._load_model()

        logger.info("PredictionService initialized.")

    # Internal Model Loader
    def _load_model(self):
        """Load champion inference pipeline from MLflow."""
        try:
            logger.info("Loading champion inference pipeline from MLflow...")

            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)

            model_uri = f"models:/{self.mlflow_config.prod_registry_name}@champion"
            logger.info(f"Model URI: {model_uri}")

            model = mlflow.pyfunc.load_model(model_uri)

            # Try to extract version
            try:
                info = mlflow.models.get_model_info(model_uri)
                version = info.model_version
                logger.info(f"Loaded model version: {self.model_version}")
            except Exception as e:
                logger.warning(f"Could not read model version: {e}")
                version = "unknown"
            logger.info("Inference pipeline loaded successfully.")
            return model, version

        except Exception as e:
            logger.error(f"Error while loading model: {e}")
            raise CustomException(e, sys)

    # Prediction
    def predict(self, input_data: pd.DataFrame) -> dict:
        """
        Apply the inference pipeline (preprocessor + model) to the input data.

        Returns:
            dict: predictions + metadata
        """
        try:
            logger.info(f"Prediction requested for {len(input_data)} samples.")

            # The sklearn pipeline inside MLflow handles preprocessing
            predictions = self.model.predict(input_data)
            predictions_list = predictions.tolist()

            response = {
                "predictions": predictions_list,
                "model_version": self.model_version,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(predictions_list),
            }

            logger.info(
                f"Prediction completed. Samples: {len(predictions_list)}, "
                f"Model version: {self.model_version}"
            )

            return response

        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise CustomException(e, sys)
