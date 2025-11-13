import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
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
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        self.client = MlflowClient()

    def evaluate_model(self, model, X, y) -> float:
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
        roc_auc = roc_auc_score(y, y_pred)
        return roc_auc

    def log_and_register_model(
        self, model, X_train, X_test, data_transformation_artifact
    ):
        input_example = pd.DataFrame(
            X_test[:5], columns=data_transformation_artifact.feature_names
        ).astype(np.float64)
        sig_sample_X = pd.DataFrame(
            X_train[:5], columns=data_transformation_artifact.feature_names
        ).astype(np.float64)
        if hasattr(model, "predict_proba"):
            sig_sample_y = model.predict_proba(sig_sample_X)[:, 1]
        else:
            sig_sample_y = model.predict(sig_sample_X)
        signature = infer_signature(sig_sample_X, sig_sample_y)
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature,
            registered_model_name=self.config.registry_name,
        )
        return model_info

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

            with mlflow.start_run(run_name=f"{self.config.model_name}_Training"):
                mlflow.set_tag("developer", "Pedro")
                mlflow.set_tag("phase", "model_training")

                # Log training parameters and context
                mlflow.log_params(self.config.best_params)
                mlflow.log_artifact(str(data_transformation_artifact.preprocessor_path))
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

                mlflow.log_metrics(
                    {"train_roc_auc": train_roc_auc, "test_roc_auc": test_roc_auc}
                )

                # Log and register the model to MLflow
                model_info = self.log_and_register_model(
                    model, X_train, X_test, data_transformation_artifact
                )

                # The version is in the model_info
                version = model_info.registered_model_version

                # Set governance tag
                self.client.set_model_version_tag(
                    name=self.config.registry_name,
                    version=version,
                    key="validation_status",
                    value="pending",
                )

                # Set alias for easy reference
                self.client.set_registered_model_alias(
                    name=self.config.registry_name,
                    alias="candidate",
                    version=version,
                )

                logger.info(
                    f"Registered model version {version} tagged as pending validation."
                )

                # Save the trained model locally
                logger.info("Saving the trained model locally.")
                joblib.dump(model, self.config.trained_model_path)

            logger.info("Model training process completed.")
            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                metric_score=test_roc_auc,
                model_name=self.config.model_name,
            )
        except Exception as e:
            raise CustomException(e, sys)
