import sys
from datetime import datetime

import mlflow
import pandas as pd

from churn_project.entity.config_entity import ModelPredictorConfig
from churn_project.exception import CustomException
from churn_project.logger import logger


class ModelPredictor:
    """
    Production-ready model predictor that loads the champion inference pipeline
    from the MLflow Model Registry and performs predictions on new data.

    Designed to be used as a singleton inside FastAPI applications.
    """

    def __init__(self, config: ModelPredictorConfig):
        """
        Initialize and store configuration.

        Args:
            config: ModelPredictorConfig containing MLflow configuration
        """
        self.config = config
        self.mlflow_config = config.mlflow_config
        self._model = None
        self._model_version = None

        logger.info("ModelPredictor initialized.")

    # Lazy Loading (loads model on first access)
    @property
    def model(self):
        """Return the loaded inference pipeline, loading it on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    # Internal Model Loader
    def _load_model(self):
        """Load champion inference pipeline from MLflow."""
        try:
            logger.info("Loading champion inference pipeline from MLflow...")

            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)

            model_uri = f"models:/{self.mlflow_config.prod_registry_name}@champion"
            logger.info(f"Model URI: {model_uri}")

            # Load sklearn pipeline wrapped by MLflow PyFunc
            model = mlflow.pyfunc.load_model(model_uri)

            # Try to extract version
            try:
                info = mlflow.models.get_model_info(model_uri)
                self._model_version = info.model_version
                logger.info(f"Loaded model version: {self._model_version}")
            except Exception as e:
                logger.warning(f"Could not read model version: {e}")
                self._model_version = "unknown"

            logger.info("Inference pipeline loaded successfully.")
            return model

        except Exception as e:
            logger.error(f"Error while loading model: {e}")
            raise CustomException(e, sys)

    # Input Validation
    def _validate_input(self, input_data: pd.DataFrame) -> None:
        """Basic input validation."""
        try:
            if input_data.empty:
                raise ValueError("Input DataFrame is empty.")

            if input_data.isnull().all().all():
                raise ValueError("Input DataFrame contains only null values.")

            logger.info(f"Input validation passed. Shape: {input_data.shape}")

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
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

            self._validate_input(input_data)

            # The sklearn pipeline inside MLflow handles preprocessing
            predictions = self.model.predict(input_data)
            predictions_list = predictions.tolist()

            response = {
                "predictions": predictions_list,
                "model_version": self._model_version,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(predictions_list),
            }

            logger.info(
                f"Prediction completed. Samples: {len(predictions_list)}, "
                f"Model version: {self._model_version}"
            )

            return response

        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise CustomException(e, sys)
