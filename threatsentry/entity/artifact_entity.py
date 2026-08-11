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
    
@dataclass
class DataValidationArtifact:
    """
    Output of Data Validation stage.
    Tells Data Transformation whether data passed validation,
    where valid/invalid files are, and where drift report is saved.
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str  
    invalid_test_file_path: str     
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    """
    Output of Data Transformation stage.
    Tells Model Trainer where transformed numpy arrays
    and the fitted preprocessor pickle are saved.
    """
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    
@dataclass
class ClassificationMetricArtifact:
    """
    Stores F1, Precision, Recall for one model evaluation.
    Tracked separately for train and test to detect overfitting.
    """
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Output of Model Trainer stage.
    Contains path to saved model and
    metrics for both train and test sets.
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact