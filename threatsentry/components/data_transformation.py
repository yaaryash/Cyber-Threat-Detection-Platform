"""
Data Transformation Component
Handles missing value imputation using KNN Imputer,
separates features from target, and saves transformed
arrays ready for model training.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from threatsentry.constant.training_pipeline import TARGET_COLUMN
from threatsentry.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from threatsentry.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from threatsentry.entity.config_entity import DataTransformationConfig
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger
from threatsentry.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logger.info("DataTransformation initialized")
        except Exception as e:
            raise ThreatDetectionException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates a KNN Imputer pipeline.
        KNN is used instead of simple mean/median because
        phishing features are correlated — a record's nearest
        neighbors give a smarter fill than a column-wide average.
        Params are defined in constants so they're easy to tune.
        """
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logger.info(f"KNNImputer initialized with params: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")

            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            logger.info(f"Train features shape: {input_feature_train_df.shape}")
            logger.info(f"Test features shape: {input_feature_test_df.shape}")

            # Fit on train only — never fit on test data
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test = preprocessor_object.transform(input_feature_test_df)

            logger.info("KNN imputation complete on train and test data")

            # Combine features + target back into arrays
            train_arr = np.c_[transformed_input_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test, np.array(target_feature_test_df)]

            # Save transformed arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # Save preprocessor — needed at inference time in app.py
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
            )
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logger.info(f"Data transformation complete — artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise ThreatDetectionException(e, sys)