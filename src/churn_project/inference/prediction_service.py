import json
import os
import sys
import tempfile
import traceback
from datetime import datetime

import mlflow.pyfunc
import pandas as pd

from churn_project.aws.s3_utils import download_s3_folder
from churn_project.exception import CustomException
from churn_project.logger import logger


class PredictionService:
    """
    Loads the champion inference pipeline from fixed S3 production folder
    (download -> mlflow.pyfunc.load_model) and performs predictions on new data.

    Designed to be instantiated once (singleton) inside FastAPI applications.
    """

    def __init__(self, prod_s3_uri: str):
        self.prod_s3_uri = prod_s3_uri
        self.model = None
        self._model_version = None
        try:
            self._load_model_from_s3()
        except Exception:
            logger.warning("No model found yet. FastAPI will run without predictions.")
        logger.info("PredictionService initialized.")

    @property
    def model_version(self):
        return self._model_version

    def _load_model_from_s3(self):
        """Download model from S3 and load it using MLflow."""
        try:
            logger.info(f"Downloading model from S3: {self.prod_s3_uri}")
            # Download to a temporary directory and load model using mlflow.pyfunc
            with tempfile.TemporaryDirectory() as temp_dir:
                download_s3_folder(self.prod_s3_uri, temp_dir)

                # Read metadata to get model version
                metadata_path = os.path.join(temp_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        self._model_version = metadata.get("version", "unknown")
                    except Exception as e:
                        logger.warning(f"Could not read metadata.json: {e}")
                        self._model_version = "unknown"
                else:
                    logger.warning("metadata.json not found in S3 model folder.")
                    self._model_version = "unknown"

                model_path = os.path.join(temp_dir, "model")
                logger.info(f"Loading model from {model_path}")
                self.model = mlflow.pyfunc.load_model(model_path)
                logger.info(f"Model loaded. Version: {self._model_version}")

        except Exception as e:
            logger.error(f"Error loading model from S3: {e}")
            logger.error(traceback.format_exc())
            raise CustomException(e, sys)

    # Prediction
    def predict(self, input_data: pd.DataFrame) -> dict:
        """
        Apply the inference pipeline (preprocessor + model) to the input data.

        Returns:
            dict: predictions + metadata
        """
        try:
            if self.model is None:
                raise Exception("Model is not loaded.")
            logger.info(f"Prediction requested for {len(input_data)} samples.")

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

        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            logger.error(traceback.format_exc())
            raise CustomException(e, sys)
