"""
NetworkModel — Custom estimator that wraps
the preprocessor and trained model together.
Used at inference time in app.py so prediction
is always consistent with training transformation.
"""
import sys

from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger


class NetworkModel:
    """
    Wraps preprocessor + model into one object.
    Ensures the same KNN imputation applied during
    training is always applied before prediction.
    """

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            logger.info("NetworkModel initialized with preprocessor and model")
        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise ThreatDetectionException(e, sys)