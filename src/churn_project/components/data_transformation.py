import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_project.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from churn_project.entity.config_entity import DataTransformationConfig
from churn_project.exception import CustomException
from churn_project.logger import logger
from churn_project.utils import save_object


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""

    def __init__(self, drop_columns: list):
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_eng = X.copy()

        try:
            X_eng["Activity_Growth"] = (
                X_eng["Total_Amt_Chng_Q4_Q1"]
                / X_eng["Total_Trans_Amt"].replace(0, np.nan)
            ).fillna(0)

            X_eng["Customer_Value"] = (
                X_eng["Total_Revolving_Bal"]
                * X_eng["Avg_Utilization_Ratio"]
                * X_eng["Credit_Limit"]
            )

        except Exception as e:
            raise CustomException(e, sys)

        X_eng.drop(columns=self.drop_columns, inplace=True)
        return X_eng


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_preprocessor(self) -> Pipeline:
        """
        Creates the preprocessing pipeline with feature engineering and scaling.
        """
        try:
            logger.info("Creating preprocessing pipeline.")
            preprocessor = Pipeline(
                steps=[
                    (
                        "feature_engineer",
                        FeatureEngineer(drop_columns=self.config.drop_columns),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        """
        Executes data transformation:
        - Loads raw train/test data
        - Applies feature engineering and scaling
        - Balances training data using SMOTE
        - Saves transformed data and preprocessor
        - Returns DataTransformationArtifact
        """
        try:
            logger.info("Starting data transformation process")

            train_df = pd.read_csv(data_ingestion_artifact.training_file_path)
            test_df = pd.read_csv(data_ingestion_artifact.testing_file_path)

            logger.info("Read train and test data completed")

            target_column = self.config.target_column

            # Split into X and y
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column].map(
                {"Attrited Customer": 1, "Existing Customer": 0}
            )
            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column].map(
                {"Attrited Customer": 1, "Existing Customer": 0}
            )

            logger.info("Split into features and target completed")

            preprocessor = self.get_preprocessor()

            logger.info("Fitting and transforming training data.")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logger.info("Balancing training data using SMOTE.")
            resampler = SMOTE(random_state=self.config.random_state)
            X_train_resampled, y_train_resampled = resampler.fit_resample(
                X_train_transformed, y_train
            )

            logger.info("Saving train and test data arrays.")
            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_transformed, y_test]
            os.makedirs(Path(self.config.transformed_train_path).parent, exist_ok=True)
            os.makedirs(Path(self.config.transformed_test_path).parent, exist_ok=True)
            np.save(self.config.transformed_train_path, train_arr)
            np.save(self.config.transformed_test_path, test_arr)

            logger.info("Saving preprocessor object.")
            save_object(file_path=self.config.preprocessor_path, obj=preprocessor)

            logger.info("Data transformation process completed.")

            return DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessor_path=self.config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
