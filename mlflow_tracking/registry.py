"""
MLflow Model Registry utilities.

Provides functions for managing model versions, stages,
and loading production models.
"""

import logging
import os
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow Model Registry manager."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize the model registry.

        Args:
            tracking_uri: MLflow tracking server URI.
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri)

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ) -> str:
        """Register a model from a run.

        Args:
            run_id: The MLflow run ID containing the model.
            model_name: Name to register the model under.
            artifact_path: Path to the model within the run artifacts.

        Returns:
            The model version string.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model {model_name} version {result.version}")
        return result.version

    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> Optional[str]:
        """Get the latest version of a registered model.

        Args:
            model_name: Name of the registered model.
            stage: Optional stage filter (Staging, Production, etc.)

        Returns:
            The latest version number, or None if not found.
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                versions = self.client.get_latest_versions(model_name)

            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """Transition a model version to a new stage.

        Args:
            model_name: Name of the registered model.
            version: Version to transition.
            stage: Target stage (Staging, Production, Archived).
            archive_existing: Whether to archive existing models in the stage.
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")

    def promote_to_production(
        self,
        model_name: str,
        version: Optional[str] = None
    ):
        """Promote a model version to production.

        Args:
            model_name: Name of the registered model.
            version: Version to promote. If None, uses latest staging version.
        """
        if version is None:
            version = self.get_latest_version(model_name, stage="Staging")
            if version is None:
                version = self.get_latest_version(model_name)

        if version is None:
            raise ValueError(f"No versions found for model {model_name}")

        self.transition_model_stage(
            model_name=model_name,
            version=version,
            stage="Production",
            archive_existing=True
        )

    def load_production_model(self, model_name: str) -> Any:
        """Load the production version of a model.

        Args:
            model_name: Name of the registered model.

        Returns:
            The loaded model.
        """
        model_uri = f"models:/{model_name}/Production"
        try:
            # Try loading as XGBoost first
            model = mlflow.xgboost.load_model(model_uri)
        except Exception:
            # Fall back to sklearn
            model = mlflow.sklearn.load_model(model_uri)

        logger.info(f"Loaded production model: {model_name}")
        return model

    def load_model_version(self, model_name: str, version: str) -> Any:
        """Load a specific version of a model.

        Args:
            model_name: Name of the registered model.
            version: Version to load.

        Returns:
            The loaded model.
        """
        model_uri = f"models:/{model_name}/{version}"
        try:
            model = mlflow.xgboost.load_model(model_uri)
        except Exception:
            model = mlflow.sklearn.load_model(model_uri)

        logger.info(f"Loaded model: {model_name} v{version}")
        return model

    def get_model_info(self, model_name: str, version: str) -> dict:
        """Get information about a model version.

        Args:
            model_name: Name of the registered model.
            version: Version to get info for.

        Returns:
            Dictionary of model information.
        """
        model_version = self.client.get_model_version(model_name, version)
        run = self.client.get_run(model_version.run_id)

        return {
            "name": model_name,
            "version": version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "description": model_version.description,
            "source": model_version.source,
            "tags": dict(model_version.tags) if model_version.tags else {},
            "run_params": run.data.params if run else {},
            "run_metrics": run.data.metrics if run else {},
        }

    def list_model_versions(self, model_name: str) -> list[dict]:
        """List all versions of a model.

        Args:
            model_name: Name of the registered model.

        Returns:
            List of version information dictionaries.
        """
        versions = []
        for mv in self.client.search_model_versions(f"name='{model_name}'"):
            versions.append({
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "creation_timestamp": mv.creation_timestamp,
            })
        return sorted(versions, key=lambda x: int(x["version"]), reverse=True)

    def delete_model_version(self, model_name: str, version: str):
        """Delete a model version.

        Args:
            model_name: Name of the registered model.
            version: Version to delete.
        """
        self.client.delete_model_version(model_name, version)
        logger.info(f"Deleted model version: {model_name} v{version}")

    def set_model_version_tag(
        self,
        model_name: str,
        version: str,
        key: str,
        value: str
    ):
        """Set a tag on a model version.

        Args:
            model_name: Name of the registered model.
            version: Version to tag.
            key: Tag key.
            value: Tag value.
        """
        self.client.set_model_version_tag(model_name, version, key, value)

    def update_model_description(
        self,
        model_name: str,
        version: str,
        description: str
    ):
        """Update the description of a model version.

        Args:
            model_name: Name of the registered model.
            version: Version to update.
            description: New description.
        """
        self.client.update_model_version(
            model_name, version, description=description
        )


def get_production_model(model_name: str = "fraud_detector") -> Any:
    """Convenience function to load the production model.

    Args:
        model_name: Name of the registered model.

    Returns:
        The loaded production model.
    """
    registry = ModelRegistry()
    return registry.load_production_model(model_name)
