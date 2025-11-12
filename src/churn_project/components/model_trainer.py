import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from churn_project.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from churn_project.entity.config_entity import ModelTrainerConfig
from churn_project.exception import CustomException
from churn_project.logger import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.client = MlflowClient()
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

    def evaluate_model(self, model, X, y) -> float:
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
        roc_auc = roc_auc_score(y, y_pred)
        return roc_auc

    def initiate_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training process.")

            # Load transformed data
            train_arr = np.load(data_transformation_artifact.transformed_train_path)
            test_arr = np.load(data_transformation_artifact.transformed_test_path)

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_map = {
                "XGBClassifier": XGBClassifier,
                "RandomForestClassifier": RandomForestClassifier,
            }

            model_class = model_map[self.config.model_name]

            with mlflow.start_run(run_name=f"{self.config.model_name}_Training") as run:
                mlflow.set_tag("developer", "Pedro")
                mlflow.set_tag("phase", "model_training")

                # Log training parameters and context
                mlflow.log_params(self.config.best_params)
                mlflow.log_param("expected_score", self.config.expected_score)
                mlflow.log_artifact(data_transformation_artifact.preprocessor_path)
                mlflow.log_dict(
                    {"feature_names": data_transformation_artifact.feature_names},
                    artifact_file="feature_names.json",
                )

                # Train the model
                model = model_class(**self.config.best_params)
                model.fit(X_train, y_train)

                # Evaluate the model
                train_roc_auc = self.evaluate_model(model, X_train, y_train)
                test_roc_auc = self.evaluate_model(model, X_test, y_test)

                # Validate model performance
                if test_roc_auc < self.config.expected_score:
                    mlflow.log_param("status", "rejected")
                    mlflow.log_param("reason", "Below expected_score")
                    raise Exception(
                        f"Model AUC {test_roc_auc:.4f} < Expected {self.config.expected_score}"
                    )
                else:
                    mlflow.log_param("status", "accepted")

                mlflow.log_metrics(
                    {"train_roc_auc": train_roc_auc, "test_roc_auc": test_roc_auc}
                )

                # Log model to MLflow
                input_example = pd.DataFrame(
                    X_test[:5], columns=data_transformation_artifact.feature_names
                )
                mlflow.sklearn.log_model(
                    model, name="model", input_example=input_example
                )

                # Register the model in MLflow Model Registry
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri, self.config.model_registry_name
                )
                version = registered_model.version

                # Tag metadata for governance
                self.client.set_model_version_tag(
                    name=self.config.model_registry_name,
                    version=version,
                    key="status",
                    value="trained",
                )
                self.client.set_model_version_tag(
                    name=self.config.model_registry_name,
                    version=version,
                    key="validation_status",
                    value="pending",
                )
                logger.info(
                    f"Registered model version {version} tagged as pending validation."
                )
                # Under the same name, mlflow assigns version numbers automatically

                # Save the trained model locally
                logger.info("Saving the trained model locally.")
                joblib.dump(model, self.config.trained_model_path)

            logger.info("Model training process completed.")
            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                metric_score=test_roc_auc,
                model_name=self.config.model_name,
                model_registry_version=version,
            )
        except Exception as e:
            raise CustomException(e, sys)
