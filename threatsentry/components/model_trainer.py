"""
Model Trainer Component
Trains multiple ML models, picks the best one using
GridSearchCV, tracks all experiments in ONE MLflow run
on DagsHub, and saves the final model + preprocessor.
"""
import os
import sys
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier

from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger
from threatsentry.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from threatsentry.entity.config_entity import ModelTrainerConfig
from threatsentry.utils.ml_utils.model.estimator import NetworkModel
from threatsentry.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from threatsentry.utils.ml_utils.metric.classification_metric import get_classification_score

# Load MLflow credentials from .env — never hardcode
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer:
    """
    Trains and evaluates multiple classification models,
    selects the best performer, tracks ALL experiments in
    ONE single MLflow run on DagsHub for easy comparison,
    and saves the final model artifact.
    """

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self._X_test = None   # stored for use in track_mlflow
            self._y_test = None   # stored for use in track_mlflow
            logger.info("ModelTrainer initialized")
        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def track_mlflow(self, models: dict, model_report: dict,
                     best_model_name: str,
                     classification_train_metric,
                     classification_test_metric):
        """
        Logs ALL models in ONE single MLflow run.
        Each model's metrics are prefixed with its name
        so you can compare everything on DagsHub in one place.

        DagsHub will show:
        - Random Forest_f1, Decision Tree_f1, XGBoost_f1 etc.
        - best_model tag
        - train vs test metrics for best model
        - best model artifact
        """
        try:
            mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))
            tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name="All_Models_Comparison"):

                # Tag for easy identification
                mlflow.set_tag("best_model", best_model_name)
                mlflow.set_tag("project", "CyberThreatDetectionPlatform")

                # Log every model's metrics with name prefix
                for model_name, model in models.items():
                    metric = get_classification_score(
                        y_true=self._y_test,
                        y_pred=model.predict(self._X_test)
                    )
                    # Prefix with model name — all visible in one run
                    mlflow.log_metric(
                        f"{model_name}_f1", metric.f1_score
                    )
                    mlflow.log_metric(
                        f"{model_name}_precision", metric.precision_score
                    )
                    mlflow.log_metric(
                        f"{model_name}_recall", metric.recall_score
                    )
                    mlflow.log_metric(
                        f"{model_name}_test_score", model_report[model_name]
                    )

                # Log best model summary separately
                mlflow.log_param("best_model_name", best_model_name)

                # Train metrics
                mlflow.log_metric(
                    "best_train_f1", classification_train_metric.f1_score
                )
                mlflow.log_metric(
                    "best_train_precision", classification_train_metric.precision_score
                )
                mlflow.log_metric(
                    "best_train_recall", classification_train_metric.recall_score
                )

                # Test metrics
                mlflow.log_metric(
                    "best_test_f1", classification_test_metric.f1_score
                )
                mlflow.log_metric(
                    "best_test_precision", classification_test_metric.precision_score
                )
                mlflow.log_metric(
                    "best_test_recall", classification_test_metric.recall_score
                )

                # Overfitting gap
                mlflow.log_metric(
                    "overfitting_gap",
                    abs(classification_train_metric.f1_score -
                        classification_test_metric.f1_score)
                )

                # Log best model artifact
                best_model = models[best_model_name]

                if isinstance(best_model, XGBClassifier):
                    if tracking_url_type != "file":
                        mlflow.xgboost.log_model(
                            best_model, "best_model",
                            registered_model_name=best_model_name
                        )
                    else:
                        mlflow.xgboost.log_model(best_model, "best_model")
                else:
                    if tracking_url_type != "file":
                        mlflow.sklearn.log_model(
                            best_model, "best_model",
                            registered_model_name=best_model_name
                        )
                    else:
                        mlflow.sklearn.log_model(best_model, "best_model")

            logger.info(
                "All models tracked in single MLflow run — "
                f"Best: {best_model_name} | "
                f"Test F1: {classification_test_metric.f1_score:.4f}"
            )

        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def train_model(self, X_train, y_train,
                    X_test, y_test) -> ModelTrainerArtifact:
        """
        Trains all candidate models with GridSearchCV,
        picks best by test score, checks overfitting,
        tracks everything in one MLflow run,
        saves model + preprocessor.
        """
        try:
            # Store for use in track_mlflow
            self._X_test = X_test
            self._y_test = y_test

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost": XGBClassifier(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7]
                }
            }

            logger.info("Starting model evaluation with GridSearchCV")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Pick best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Check minimum accuracy threshold
            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Best model score {best_model_score:.4f} is below "
                    f"expected accuracy "
                    f"{self.model_trainer_config.expected_accuracy}"
                )

            # Log all model results to terminal
            logger.info("=" * 40)
            logger.info("All Model Results:")
            logger.info("=" * 40)
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                metric = get_classification_score(
                    y_true=y_test, y_pred=y_pred
                )
                logger.info(
                    f"{model_name:<25} | "
                    f"F1: {metric.f1_score:.4f} | "
                    f"Precision: {metric.precision_score:.4f} | "
                    f"Recall: {metric.recall_score:.4f} | "
                    f"{'⭐ BEST' if model_name == best_model_name else ''}"
                )
            logger.info("=" * 40)
            logger.info(
                f"Best model: {best_model_name} | "
                f"Score: {best_model_score:.4f}"
            )
            logger.info("=" * 40)

            # Final metrics for best model train + test
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )

            # Check overfitting
            diff = abs(
                classification_train_metric.f1_score -
                classification_test_metric.f1_score
            )
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logger.warning(
                    f"Possible overfitting — "
                    f"Train F1: {classification_train_metric.f1_score:.4f} | "
                    f"Test F1: {classification_test_metric.f1_score:.4f} | "
                    f"Diff: {diff:.4f}"
                )
            else:
                logger.info(
                    f"No overfitting — "
                    f"Train F1: {classification_train_metric.f1_score:.4f} | "
                    f"Test F1: {classification_test_metric.f1_score:.4f} | "
                    f"Diff: {diff:.4f}"
                )

            # Single MLflow run tracking everything
            self.track_mlflow(
                models=models,
                model_report=model_report,
                best_model_name=best_model_name,
                classification_train_metric=classification_train_metric,
                classification_test_metric=classification_test_metric
            )

            # Save best model
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )
            save_object("final_model/model.pkl", best_model)
            logger.info("Best model saved to artifacts and final_model/")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise ThreatDetectionException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Entry point — loads transformed numpy arrays
        and kicks off model training.
        """
        try:
            logger.info("Starting model training")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            # Last column is target, rest are features
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logger.info(
                f"Train shape: {X_train.shape} | "
                f"Test shape: {X_test.shape}"
            )

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise ThreatDetectionException(e, sys)