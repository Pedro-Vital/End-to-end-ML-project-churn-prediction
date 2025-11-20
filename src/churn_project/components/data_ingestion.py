import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from churn_project.entity.artifact_entity import DataIngestionArtifact
from churn_project.entity.config_entity import DataIngestionConfig
from churn_project.exception import CustomException
from churn_project.logger import logger
from churn_project.utils import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def fetch_and_save_data(self) -> None:
        """Fetch data from MySQL and save directly to CSV"""
        try:
            logger.info("Starting data ingestion process")

            os.makedirs(self.config.raw_data_path.parent, exist_ok=True)

            # Build SQLAlchemy engine
            engine = create_engine(
                f"mysql+pymysql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}/{self.config.db_name}"
            )

            columns = ", ".join(self.config.columns.keys())
            query = self.config.base_query.format(columns)
            logger.info(f"Executing query: {query}")

            df = pd.read_sql(query, con=engine)

            if df.empty:
                logger.warning("Fetched dataset is empty — skipping save.")
                return

            logger.info(
                f"Data fetched successfully: {df.shape[0]} rows, {df.shape[1]} columns"
            )

            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(
                f"Data saved to {self.config.raw_data_path} "
                f"({get_size(self.config.raw_data_path)})"
            )

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)

    def split_data(self) -> None:
        """Split data into training and testing sets"""
        try:

            logger.info("Reading data from raw data file for splitting")
            df = pd.read_csv(self.config.raw_data_path)

            logger.info("Splitting data into training and testing sets")
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=self.config.random_state,
            )

            # Save training data
            os.makedirs(self.config.training_path.parent, exist_ok=True)
            train_df.to_csv(self.config.training_path, index=False)
            logger.info(
                f"Training data saved to {self.config.training_path} with size {get_size(self.config.training_path)}"
            )

            # Save testing data
            os.makedirs(self.config.testing_path.parent, exist_ok=True)
            test_df.to_csv(self.config.testing_path, index=False)
            logger.info(
                f"Testing data saved to {self.config.testing_path} with size {get_size(self.config.testing_path)}"
            )

        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Initiate data ingestion process"""
        self.fetch_and_save_data()
        self.split_data()

        return DataIngestionArtifact(
            training_path=self.config.training_path,
            testing_path=self.config.testing_path,
            raw_data_path=self.config.raw_data_path,
        )
