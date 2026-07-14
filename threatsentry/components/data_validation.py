import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from threatsentry.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from threatsentry.entity.config_entity import DataValidationConfig
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger
from threatsentry.constant.training_pipeline import SCHEMA_FILE_PATH
from threatsentry.utils.main_utils.utils import read_yaml_file


class DataValidation:
    """
    Validates that ingested data:
    1. Has the correct number of columns (schema check)
    2. Has no significant statistical drift between
       train and test distributions (KS test)
    """

    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info("DataValidation initialized")
        except Exception as e:
            raise ThreatDetectionException(e, sys)