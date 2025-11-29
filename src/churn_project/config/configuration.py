import os
from pathlib import Path

from churn_project.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from churn_project.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    MlflowConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
)
from churn_project.utils import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_mlflow_config(self) -> MlflowConfig:
        mlflow_config = self.config.mlflow

        mlflow_configuration = MlflowConfig(
            tracking_uri=mlflow_config.tracking_uri,
            experiment_name=mlflow_config.experiment_name,
            registry_name=mlflow_config.registry_name,
            prod_registry_name=mlflow_config.prod_registry_name,
        )
        return mlflow_configuration

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion  # dict-like ConfigBox
        columns = self.schema.columns  # dict-like ConfigBox

        create_directories([config.root_dir])

        # Prefer environment variables if available for sensitive info
        db_host = os.getenv("DB_HOST", config.db_host)
        db_user = os.getenv("DB_USER", config.db_user)
        db_password = os.getenv("DB_PASSWORD", config.db_password)
        db_name = os.getenv("DB_NAME", config.db_name)

        data_ingestion_config = DataIngestionConfig(
            db_host=db_host,
            db_user=db_user,
            db_password=db_password,
            db_name=db_name,
            base_query=config.base_query,
            raw_data_path=Path(config.raw_data_path),
            training_path=Path(config.training_path),
            testing_path=Path(config.testing_path),
            train_test_split_ratio=config.train_test_split_ratio,
            random_state=config.random_state,
            columns=columns,
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        columns = self.schema.columns

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            validation_report_path=Path(config.validation_report_path),
            columns=columns,
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            transformed_train_path=Path(config.transformed_train_path),
            transformed_test_path=Path(config.transformed_test_path),
            preprocessor_path=Path(config.preprocessor_path),
            target_column=self.schema.target_column,  # target str
            drop_columns=self.schema.drop_columns,  # list of str
            random_state=config.random_state,
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        mlflow_config = self.get_mlflow_config()

        model_trainer_config = ModelTrainerConfig(
            model_name=config.model_name,
            target_column=self.schema.target_column,
            best_params=params.get(config.model_name, {}),
            mlflow_config=mlflow_config,
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        mlflow_config = self.get_mlflow_config()

        model_evaluation_config = ModelEvaluationConfig(
            change_threshold=config.change_threshold,
            target_column=self.schema.target_column,
            model_evaluation_report_path=Path(config.model_evaluation_report_path),
            mlflow_config=mlflow_config,
        )
        return model_evaluation_config

    def get_model_pusher_config(self) -> ModelPusherConfig:
        mlflow_config = self.get_mlflow_config()

        model_pusher_config = ModelPusherConfig(
            mlflow_config=mlflow_config,
        )
        return model_pusher_config
