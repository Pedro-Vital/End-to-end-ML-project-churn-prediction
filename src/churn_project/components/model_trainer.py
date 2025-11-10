import sys

import mlflow
import numpy as np
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
from churn_project.utils import save_object


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Churn_Production_Training")
        mlflow.sklearn.autolog()
        mlflow.xgboost.autolog()

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

                mlflow.log_params(self.config.best_params)

                model = model_class(**self.config.best_params)
                model.fit(X_train, y_train)

                y_train_pred = model.predict_proba(X_train)[:, 1]
                y_test_pred = model.predict_proba(X_test)[:, 1]

                train_roc_auc = roc_auc_score(y_train, y_train_pred)
                test_roc_auc = roc_auc_score(y_test, y_test_pred)

                if test_roc_auc < self.config.expected_score:
                    mlflow.log_param("status", "rejected")
                    mlflow.log_param("reason", "Below expected_score")
                    raise Exception(
                        f"Model AUC {test_roc_auc:.4f} < Expected {self.config.expected_score}"
                    )
                else:
                    mlflow.log_param("status", "accepted")

                mlflow.log_metric("train_roc_auc", train_roc_auc)
                mlflow.log_metric("test_roc_auc", test_roc_auc)

                # mlflow.sklearn.log_model(model, name=f"{self.config.model_name}_model")
                # I will be just saving the model locally instead of logging to mlflow

                save_object(file_path=self.config.trained_model_path, obj=model)

            logger.info("Model training process completed.")
            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                metric_score=test_roc_auc,
                model_name=self.config.model_name,
            )
        except Exception as e:
            raise CustomException(e, sys)
