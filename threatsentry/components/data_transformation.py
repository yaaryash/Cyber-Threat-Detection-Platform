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