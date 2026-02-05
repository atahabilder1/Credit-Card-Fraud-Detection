"""
MLflow experiment tracking utilities.

Provides functions for logging training experiments, metrics,
parameters, and artifacts.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow experiment tracking manager."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "fraud-detection"
    ):
        """Initialize the experiment tracker.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the experiment to use.
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.experiment_name = experiment_name
        self.client = None
        self._run_id = None

        self._setup_tracking()

    def _setup_tracking(self):
        """Set up MLflow tracking configuration."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient(self.tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created experiment: {self.experiment_name}")
            else:
                logger.info(f"Using existing experiment: {self.experiment_name}")

            mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            logger.warning(f"Failed to connect to MLflow server: {e}")
            logger.info("Running in offline mode")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None
    ) -> str:
        """Start a new MLflow run.

        Args:
            run_name: Optional name for the run.
            tags: Optional tags to attach to the run.

        Returns:
            The run ID.
        """
        if run_name is None:
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        default_tags = {
            "developer": "Anik Tahabilder",
            "project": "credit-card-fraud-detection",
        }
        if tags:
            default_tags.update(tags)

        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        self._run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_name} (ID: {self._run_id})")
        return self._run_id

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run.

        Args:
            status: Final status of the run.
        """
        if self._run_id:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self._run_id}")
            self._run_id = None

    def log_params(self, params: dict[str, Any]):
        """Log parameters to the current run.

        Args:
            params: Dictionary of parameter names and values.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for time series metrics.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """Log a model to MLflow.

        Args:
            model: The trained model object.
            artifact_path: Path within the artifact store.
            registered_model_name: If provided, register the model.
        """
        # Determine model flavor
        model_type = type(model).__name__

        if "XGB" in model_type:
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )
        elif "RandomForest" in model_type or "LogisticRegression" in model_type:
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )
        else:
            # Generic sklearn model
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )

        logger.info(f"Logged model to {artifact_path}")
        if registered_model_name:
            logger.info(f"Registered model as: {registered_model_name}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file.

        Args:
            local_path: Path to the local file.
            artifact_path: Optional subdirectory in artifact store.
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts.

        Args:
            local_dir: Path to the local directory.
            artifact_path: Optional subdirectory in artifact store.
        """
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.debug(f"Logged artifacts from: {local_dir}")

    def log_figure(self, figure: Any, artifact_file: str):
        """Log a matplotlib or plotly figure.

        Args:
            figure: The figure object.
            artifact_file: Filename for the artifact.
        """
        mlflow.log_figure(figure, artifact_file)
        logger.debug(f"Logged figure: {artifact_file}")

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run.

        Args:
            key: Tag name.
            value: Tag value.
        """
        mlflow.set_tag(key, value)

    def log_training_results(
        self,
        model: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        model_name: str = "fraud_detector"
    ):
        """Convenience method to log a complete training run.

        Args:
            model: Trained model.
            params: Training parameters.
            metrics: Evaluation metrics.
            model_name: Name for model registration.
        """
        self.log_params(params)
        self.log_metrics(metrics)
        self.log_model(model, registered_model_name=model_name)


def log_training_run(
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    run_name: Optional[str] = None,
    experiment_name: str = "fraud-detection",
    register_model: bool = True,
    model_name: str = "fraud_detector"
) -> str:
    """Convenience function to log a complete training run.

    Args:
        model: Trained model.
        params: Training parameters.
        metrics: Evaluation metrics.
        run_name: Optional name for the run.
        experiment_name: Name of the experiment.
        register_model: Whether to register the model.
        model_name: Name for model registration.

    Returns:
        The run ID.
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)
    run_id = tracker.start_run(run_name=run_name)

    try:
        tracker.log_params(params)
        tracker.log_metrics(metrics)

        if register_model:
            tracker.log_model(model, registered_model_name=model_name)
        else:
            tracker.log_model(model)

        tracker.end_run(status="FINISHED")
    except Exception as e:
        tracker.end_run(status="FAILED")
        raise e

    return run_id
