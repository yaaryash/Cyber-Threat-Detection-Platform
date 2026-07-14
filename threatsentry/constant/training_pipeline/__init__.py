import os
import sys
import numpy as np
import pandas as pd

# --- Core pipeline settings ---
TARGET_COLUMN = "Result"
PIPELINE_NAME: str = "CyberThreatDetectionPlatform"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "phishing_data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")


# --- Data Ingestion constants ---
DATA_INGESTION_COLLECTION_NAME: str = "PhishingData"
DATA_INGESTION_DATABASE_NAME: str = "CyberThreatDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


# --- Data Validation constants ---
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"