from pathlib import Path

from churn_project.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from churn_project.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
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

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion  # dict-like ConfigBox
        columns = self.schema.columns  # dict-like ConfigBox

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            db_host=config.db_host,
            db_user=config.db_user,
            db_password=config.db_password,
            db_name=config.db_name,
            base_query=config.base_query,
            feature_store_file_path=Path(config.feature_store_file_path),
            training_file_path=Path(config.training_file_path),
            testing_file_path=Path(config.testing_file_path),
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
            root_dir=Path(config.root_dir),
            validation_report_path=Path(config.validation_report_path),
            columns=columns,
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        target_column = self.schema.target_column  # target str
        drop_columns = self.schema.drop_columns  # list of str

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            transformed_train_path=Path(config.transformed_train_path),
            transformed_test_path=Path(config.transformed_test_path),
            preprocessor_path=Path(config.preprocessor_path),
            target_column=target_column,
            drop_columns=drop_columns,
            random_state=config.random_state,
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        mlflow_config = self.config.mlflow
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            model_name=config.model_name,
            registry_name=mlflow_config.registry_name,
            best_params=params.get(config.model_name, {}),
            mlflow_uri=mlflow_config.tracking_uri,
            mlflow_experiment_name=mlflow_config.training_experiment_name,
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        mlflow_config = self.config.mlflow

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_evaluation_report_path=Path(config.model_evaluation_report_path),
            registry_name=mlflow_config.registry_name,
            prod_registry_name=mlflow_config.prod_registry_name,
            mlflow_uri=mlflow_config.tracking_uri,
            mlflow_experiment_name=mlflow_config.evaluation_experiment_name,
        )
        return model_evaluation_config
