import yaml
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger
import os,sys
import numpy as np

"""
Utility functions used across the entire pipeline.
All generic reusable code lives here
"""
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ThreatDetectionException(e, sys) from e