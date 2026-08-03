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
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
            raise ThreatDetectionException(e, sys) from e