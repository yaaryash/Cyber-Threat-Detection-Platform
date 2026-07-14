import os
import sys
import numpy as np
import pandas as pd
import pymongo
import certifi
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger
from threatsentry.entity.config_entity import DataIngestionConfig
from threatsentry.entity.artifact_entity import DataIngestionArtifact

MONGO_DB_URI = os.getenv("MONGO_DB_URI")

class DataIngestion:
    """
    Handles the full data ingestion pipeline:
    1. Pull raw data from MongoDB
    2. Save to local feature store
    3. Split into train/test CSV files
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info("DataIngestion initialized with config")
        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Connects to MongoDB and pulls the phishing
        dataset collection as a pandas DataFrame.
        Removes MongoDB's internal _id column and
        replaces 'na' strings with proper NaN values.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logger.info(f"Connecting to MongoDB — DB: {database_name}, Collection: {collection_name}")

            ca = certifi.where()
            mongo_client = pymongo.MongoClient(MONGO_DB_URI, tlsCAFile=ca)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            logger.info(f"Fetched {len(df)} records from MongoDB")

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            logger.info(f"DataFrame shape after cleaning: {df.shape}")

            return df

        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Saves the raw DataFrame to the feature store
        as a CSV file for reproducibility — so we always
        know exactly what raw data was used in this run.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"Raw data saved to feature store: {feature_store_file_path}")

            return dataframe

        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Splits the DataFrame into train and test sets
        using the ratio defined in constants (default 80/20),
        and saves both as separate CSV files.
        """
        try:
            logger.info(f"Splitting data — test ratio: {self.data_ingestion_config.train_test_split_ratio}")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42    # ← added: reproducible splits every run
            )

            logger.info(f"Train size: {len(train_set)} | Test size: {len(test_set)}")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logger.info("Train and test files saved successfully")

        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Master method — runs all 3 steps in sequence
        and returns a DataIngestionArtifact with paths
        to the train and test files for the next stage.
        """
        try:
            logger.info("Starting data ingestion")

            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logger.info(f"Data ingestion complete — artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise ThreatDetectionException(e, sys)