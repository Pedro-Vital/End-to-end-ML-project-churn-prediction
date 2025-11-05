import csv
import os
import sys
from churn_project.exception import CustomException

import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split

from churn_project.entity.artifact_entity import DataIngestionArtifact
from churn_project.entity.config_entity import DataIngestionConfig
from churn_project.logger import logger
from churn_project.utils import get_size




class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config



    def fetch_and_save_data(self) -> None:
        """Fetch data from MySQL and save directly to CSV"""
        try:
            logger.info("Starting data ingestion process")
            
            # Ensure the parent directory exists
            os.makedirs(self.config.feature_store_file_path.parent, exist_ok=True)

            # Connect to the database
            with pymysql.connect(
                host=self.config.db_host,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
            ) as connection:
                columns = ", ".join(self.config.columns)
                query = self.config.base_query.format(columns)
                logger.info(f"Executing query: {query}")
                
                # Fetch data directly into DataFrame
                df = pd.read_sql(query, connection)
                logger.info(f"Data fetched successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Save directly to CSV
                df.to_csv(self.config.feature_store_file_path, index=False)
                logger.info(
                    f"Data saved to {self.config.feature_store_file_path} "
                    f"with size {get_size(self.config.feature_store_file_path)}"
                )
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)



    def split_data(self) -> None:
        """Split data into training and testing sets"""
        try:

            logger.info("Reading data from feature store for splitting")
            df = pd.read_csv(self.config.feature_store_file_path)

            logger.info("Splitting data into training and testing sets")
            train_df, test_df = train_test_split(
                df, test_size=self.config.train_test_split_ratio, random_state=self.config.random_state
            )

            # Save training data
            os.makedirs(self.config.training_file_path.parent, exist_ok=True)
            train_df.to_csv(self.config.training_file_path, index=False)
            logger.info(
                f"Training data saved to {self.config.training_file_path} with size {get_size(self.config.training_file_path)}"
            )

            # Save testing data
            os.makedirs(self.config.testing_file_path.parent, exist_ok=True)
            test_df.to_csv(self.config.testing_file_path, index=False)
            logger.info(
                f"Testing data saved to {self.config.testing_file_path} with size {get_size(self.config.testing_file_path)}"
            )

        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise CustomException(e, sys)



    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Initiate data ingestion process"""
        self.fetch_and_save_data()
        self.split_data()

        return DataIngestionArtifact(
            training_file_path=self.config.training_file_path,
            testing_file_path=self.config.testing_file_path,
            feature_store_file_path=self.config.feature_store_file_path,
        )
