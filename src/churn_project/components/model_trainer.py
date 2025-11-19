import sys

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from churn_project.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from churn_project.entity.config_entity import ModelTrainerConfig
from churn_project.exception import CustomException
from churn_project.logger import logger
from churn_project.utils import evaluate_clf


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.mlflow_config = config.mlflow_config
        self.client = mlflow.tracking.MlflowClient(
            tracking_uri=self.config.mlflow_config.tracking_uri
        )

    def initiate_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training process.")

            # Load transformed data
            train_arr = np.load(data_transformation_artifact.transformed_train_path)

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            model_map = {
                "XGBClassifier": XGBClassifier,
                "RandomForestClassifier": RandomForestClassifier,
            }

            model_class = model_map[self.config.model_name]

            # Log training parameters and context
            mlflow.set_tag("developer", "Pedro")
            mlflow.log_params(self.config.best_params)
            mlflow.log_dict(
                {"feature_names": data_transformation_artifact.feature_names},
                artifact_file="feature_names.json",
            )

            # Log preprocessor
            preprocessor = joblib.load(data_transformation_artifact.preprocessor_path)
            mlflow.sklearn.log_model(
                preprocessor,
                artifact_path="preprocessor",
            )

            # Train the model
            model = model_class(**self.config.best_params)
            model.fit(X_train, y_train)

            # Evaluate the model on training data
            train_acc, train_f1, train_auc = evaluate_clf(model, X_train, y_train)

            # Log evaluation metrics
            mlflow.log_metrics(
                {
                    "train_accuracy": train_acc,
                    "train_f1_score": train_f1,
                    "train_roc_auc": train_auc,
                }
            )

            # Log and register the model to MLflow
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train[:5],
                registered_model_name=self.mlflow_config.registry_name,
            )

            # The version is in the model_info
            version = model_info.registered_model_version

            # Log registry version in the run
            mlflow.log_param("registry_version", version)

            logger.info(f"Registered model version {version}.")

            # Save the trained model locally just as fallback option
            logger.info("Saving the trained model locally.")
            joblib.dump(model, self.config.trained_model_path)

            logger.info("Model training process completed.")
            return ModelTrainerArtifact(
                registry_version=version,
            )
        except Exception as e:
            raise CustomException(e, sys)
