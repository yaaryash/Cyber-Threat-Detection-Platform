"""
Classification metric utility.
Computes F1, Precision and Recall scores
and returns them as a ClassificationMetricArtifact.
"""
import sys
from sklearn.metrics import f1_score, precision_score, recall_score

from threatsentry.entity.artifact_entity import ClassificationMetricArtifact
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Computes classification metrics between
    true labels and predicted labels.
    Used in model trainer to evaluate each model.
    """
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        logger.info(f"F1: {model_f1_score:.4f} | "
                    f"Precision: {model_precision_score:.4f} | "
                    f"Recall: {model_recall_score:.4f}")

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric

    except Exception as e:
        raise ThreatDetectionException(e, sys)