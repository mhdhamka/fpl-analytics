from src.clean import DataValidationError, clean_data
from src.ingest import IngestionError, save_raw_data
from src.logging_config import get_logger
from src.model import load_feature_importances, train_points_model
from src.visualize import (
    plot_feature_importance,
    plot_top_goalscorers,
    plot_value_picks,
    plot_xg_vs_actual,
)

logger = get_logger(__name__)


def run_pipeline():
    """Runs the complete pipeline end-to-end: ingest -> clean -> visualize -> train."""
    logger.info("=== Step 1/4: Ingesting data from FPL API ===")
    try:
        save_raw_data()
    except IngestionError:
        logger.exception("Ingestion failed after all retries. Aborting pipeline.")
        raise

    logger.info("=== Step 2/4: Cleaning and validating data ===")
    try:
        clean_data()
    except DataValidationError:
        logger.exception("Data failed validation. Aborting pipeline.")
        raise

    logger.info("=== Step 3/4: Generating visualizations ===")
    plot_top_goalscorers()
    plot_xg_vs_actual()
    plot_value_picks()

    logger.info("=== Step 4/4: Training and persisting the ML model ===")
    model = train_points_model()
    if model is not None:
        plot_feature_importance(load_feature_importances(model))

    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    run_pipeline()
