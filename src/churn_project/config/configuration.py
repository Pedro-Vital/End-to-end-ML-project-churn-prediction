from pathlib import Path

from churn_project.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from churn_project.entity.config_entity import DataIngestionConfig, DataValidationConfig
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
        config = self.config.data_ingestion
        columns = self.schema.columns

        create_directories([config.root_dir])  # Create artifacts/data_ingestion/

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
