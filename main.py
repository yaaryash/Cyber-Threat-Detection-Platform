"""
main.py — Pipeline entry point
Run this to trigger the full training pipeline locally.
Add stages one by one as they are built.
"""
import sys

from threatsentry.components.data_ingestion import DataIngestion
from threatsentry.components.data_validation import DataValidation
from threatsentry.components.data_transformation import DataTransformation
from threatsentry.components.model_trainer import ModelTrainer
from threatsentry.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger


if __name__ == '__main__':
    try:
        logger.info("=" * 50)
        logger.info("Training Pipeline Started")
        logger.info("=" * 50)

        # Stage 1 — Data Ingestion
        logger.info("Stage 1: Data Ingestion")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        # Stage 2 — Data Validation
        logger.info("Stage 2: Data Validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        logger.info(f"Data Validation complete — {data_validation_artifact}")

        # Stage 3 — Data Transformation
        logger.info("Stage 3: Data Transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logger.info(f"Data Transformation complete — {data_transformation_artifact}")
        
        # Stage 4 — Model Training
        logger.info("Stage 4: Model Training")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logger.info(f"Model Training complete — {model_trainer_artifact}")

        logger.info("=" * 50)
        logger.info("Pipeline run complete")
        logger.info("=" * 50)

        logger.info("Pipeline run complete")

    except Exception as e:
        raise ThreatDetectionException(e, sys)