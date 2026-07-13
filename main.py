"""
main.py — Pipeline entry point
Run this to trigger the full training pipeline locally.
Add stages one by one as they are built.
"""
import sys

from threatsentry.components.data_ingestion import DataIngestion
from threatsentry.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig
)
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger


if __name__ == '__main__':
    try:
        logger.info("=" * 50)
        logger.info("Training Pipeline Started")
        logger.info("=" * 50)

        # Step 1 — Data Ingestion
        logger.info("Stage 1: Data Ingestion")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        # Step 2 — Data Validation (coming soon)
        # Step 3 — Data Transformation (coming soon)
        # Step 4 — Model Training (coming soon)

        logger.info("Pipeline run complete")

    except Exception as e:
        raise ThreatDetectionException(e, sys)