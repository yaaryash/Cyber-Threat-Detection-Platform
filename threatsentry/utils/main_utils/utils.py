import yaml
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger
import os,sys
import numpy as np
import pickle

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

def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a numpy array to a .npy file.
    Used after data transformation to save
    processed train/test arrays.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info(f"Numpy array saved to: {file_path}")
    except Exception as e:
        raise ThreatDetectionException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Saves any Python object as a pickle file.
    Used to save trained model and preprocessor.
    """
    try:
        logger.info(f"Saving object to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved successfully")
    except Exception as e:
        raise ThreatDetectionException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a numpy array from a .npy file.
    Used in model trainer to load transformed data.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ThreatDetectionException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Loads a pickle file and returns the object.
    Used to load model and preprocessor at inference time.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise ThreatDetectionException(e, sys) from e