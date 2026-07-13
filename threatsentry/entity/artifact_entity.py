"""
Artifact dataclasses — each pipeline stage returns one of these.
It's the 'output receipt' passed to the next stage.
"""
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Output of Data Ingestion stage.
    Tells the next stage (Data Validation) where
    the train and test files are saved.
    """
    trained_file_path: str
    test_file_path: str