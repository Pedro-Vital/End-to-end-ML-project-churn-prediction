import csv
import os

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

    def fetch_data(self) -> str:
        logger.info("Starting data ingestion process")

        # Connect to the database
        connection = pymysql.connect(
            host=self.config.db_host,
            user=self.config.db_user,
            password=self.config.db_password,
            database=self.config.db_name,
            cursorclass=pymysql.cursors.DictCursor,
        )

        try:
            with connection.cursor() as cursor:
                cursor.execute(self.config.query)
                data = cursor.fetchall()
                logger.info("Data fetched successfully from MySQL database.")
                return data
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e
        finally:
            connection.close()

    def save_data_to_csv(self, data) -> None:
        try:
            csv_file_path = self.config.feature_store_file_path
            # Ensure the parent directory exists
            os.makedirs(csv_file_path.parent, exist_ok=True)

            logger.info(f"Saving data to {csv_file_path}")

            with csv_file_path.open(mode="w", newline="") as file:
                fieldnames = data[0].keys() if data else []
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write data
                writer.writerows(data)

            logger.info(
                f"Data saved to {csv_file_path} with size {get_size(csv_file_path)}"
            )
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            raise e

    def split_data(self) -> None:
        try:

            logger.info("Reading data from feature store for splitting")
            df = pd.read_csv(self.config.feature_store_file_path)

            logger.info("Splitting data into training and testing sets")
            train_df, test_df = train_test_split(
                df, test_size=self.config.train_test_split_ratio, random_state=42
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
            raise e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        data = self.fetch_data()
        self.save_data_to_csv(data)
        self.split_data()

        return DataIngestionArtifact(
            training_file_path=self.config.training_file_path,
            testing_file_path=self.config.testing_file_path,
        )
