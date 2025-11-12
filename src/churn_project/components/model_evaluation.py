import sys

import joblib
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from churn_project.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from churn_project.entity.config_entity import ModelEvaluationConfig
from churn_project.exception import CustomException
from churn_project.logger import logger
from churn_project.utils import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.client = MlflowClient()
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

    def _get_latest_production_model(self):
        try:
            logger.info("Loading production model from MLflow Model Registry.")
            model = mlflow.sklearn.load_model(
                f"models:/{self.config.model_registry_name}@live"
            )
            return model
        except Exception:
            logger.warning("No production model found in the registry.")
            return None

    def evaluate_model(self, model, X, y) -> float:
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
        roc_auc = roc_auc_score(y, y_pred)
        return roc_auc

    def initiate_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation process.")

            # Load transformed test data
            test_arr = np.load(data_transformation_artifact.transformed_test_path)
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # Load the newly trained model
            new_model = joblib.load(model_trainer_artifact.trained_model_path)

            with mlflow.start_run(run_name="Model_evaluation"):

                # Evaluate new model
                new_model_auc = self.evaluate_model(new_model, X_test, y_test)
                logger.info(f"New model AUC: {new_model_auc:.4f}")
                mlflow.log_metric("new_model_auc", new_model_auc)

                # Load production model registry
                production_model = self._get_latest_production_model()

                if production_model:
                    # Evaluate current model
                    prod_model_auc = self.evaluate_model(
                        production_model, X_test, y_test
                    )
                    mlflow.log_metric("production_model_auc", prod_model_auc)
                    logger.info(f"Production model AUC: {prod_model_auc:.4f}")
                else:
                    prod_model_auc = 0.0
                    logger.info(
                        "No production model found. Accepting new model by default."
                    )

                # Compare models and promote if better
                if new_model_auc > prod_model_auc:
                    logger.info(
                        "New model outperforms the production model. Promoting to Staging."
                    )
                    is_model_accepted = True

                    # 1. Tag model as approved in MLflow
                    self.client.set_model_version_tag(
                        name=self.config.model_registry_name,
                        version=model_trainer_artifact.model_registry_version,
                        key="validation_status",
                        value="approved",
                    )

                    # 2. Assign alias (e.g., 'candidate' or 'champion')
                    self.client.set_registered_model_alias(
                        name=self.config.model_registry_name,
                        alias="candidate",
                        version=model_trainer_artifact.model_registry_version,
                    )

                    # 3. Optionally copy to staging/prod environment
                    dst_name = self.config.model_registry_name.replace("dev", "staging")
                    new_version = self.client.copy_model_version(
                        src_model_uri=f"models:/{self.config.model_registry_name}@candidate",
                        dst_name=dst_name,
                    )
                    logger.info(
                        f"Copied version to {dst_name} as {new_version.version}"
                    )

                else:
                    logger.info(
                        "New model does not outperform the production model. Tagging as rejected."
                    )
                    is_model_accepted = False
                    self.client.set_model_version_tag(
                        name=self.config.model_registry_name,
                        version=model_trainer_artifact.model_registry_version,
                        key="validation_status",
                        value="rejected",
                    )

                evaluation_report = {
                    "new_model_auc": new_model_auc,
                    "production_model_auc": prod_model_auc,
                    "is_model_accepted": is_model_accepted,
                }

                # Save evaluation report
                save_json(self.config.model_evaluation_report_path, evaluation_report)
                mlflow.log_dict(evaluation_report, "evaluation_report.json")
                logger.info(
                    f"Model evaluation report saved at {self.config.model_evaluation_report_path}"
                )

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                model_evaluation_report_path=self.config.model_evaluation_report_path,
            )
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)
