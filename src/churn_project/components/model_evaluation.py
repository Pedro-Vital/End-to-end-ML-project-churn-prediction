import sys

import mlflow
import pandas as pd

from churn_project.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from churn_project.entity.config_entity import ModelEvaluationConfig
from churn_project.exception import CustomException
from churn_project.logger import logger
from churn_project.utils import evaluate_clf


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.mlflow_config = config.mlflow_config
        self.client = mlflow.tracking.MlflowClient(
            tracking_uri=self.config.mlflow_config.tracking_uri
        )

    def load_models(self, model_trainer_artifact: ModelTrainerArtifact):
        logger.info("Loading models for evaluation.")
        # Load production model registry
        try:
            production_model = mlflow.sklearn.load_model(
                f"models:/{self.mlflow_config.prod_registry_name}@champion"
            )
            logger.info("Production model loaded successfully.")
        except Exception:
            production_model = None
            logger.warning(
                "No production model found. This might be the first deployment."
            )
        # Load newly trained model
        try:
            # Use version-style URI for the newly trained model
            new_model_uri = f"models:/{self.mlflow_config.registry_name}/{model_trainer_artifact.registry_version}"
            new_model = mlflow.sklearn.load_model(new_model_uri)
            logger.info("Newly trained model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load newly trained model.")
            raise CustomException(e, sys)
        return production_model, new_model

    def initiate_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation process.")

            # Load raw test data
            test_df = pd.read_csv(data_transformation_artifact.raw_test_path)
            X_test = test_df.drop(columns=[self.config.target_column], axis=1)
            y_test = test_df[self.config.target_column].map(
                {"Attrited Customer": 1, "Existing Customer": 0}
            )

            # Load models for evaluation
            production_model, new_model = self.load_models(model_trainer_artifact)

            if production_model:
                # Evaluate current production model
                prod_acc, prod_f1, prod_auc = evaluate_clf(
                    production_model, X_test, y_test
                )
                logger.info(f"Production model AUC: {prod_auc:.4f}")
            else:
                prod_acc, prod_f1, prod_auc = 0, 0, 0
                logger.info("Accepting new model by default.")

            # Evaluate newly trained model
            new_acc, new_f1, new_auc = evaluate_clf(new_model, X_test, y_test)
            logger.info(f"New model AUC: {new_auc:.4f}")

            # Compare models adding epsilon to avoid noise issues
            is_model_accepted = new_auc > (prod_auc + self.config.change_threshold)
            if is_model_accepted:
                logger.info("New model outperforms the production model.")
            else:
                logger.info("New model does not outperform the production model.")

            evaluation_report = {
                "production_model_accuracy": prod_acc,
                "new_model_accuracy": new_acc,
                "production_model_f1_score": prod_f1,
                "new_model_f1_score": new_f1,
                "production_model_auc": prod_auc,
                "new_model_auc": new_auc,
                "is_model_accepted": is_model_accepted,
            }

            # Log evaluation report and metrics to MLflow
            mlflow.log_dict(evaluation_report, "evaluation_report.json")
            mlflow.log_metrics(
                {
                    "test_accuracy": new_acc,
                    "test_f1_score": new_f1,
                    "test_roc_auc": new_auc,
                },
                run_id=mlflow.active_run().info.run_id,
            )
            mlflow.set_tag("is_model_accepted", str(is_model_accepted))

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                model_evaluation_report_path=self.config.model_evaluation_report_path,
            )
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)
