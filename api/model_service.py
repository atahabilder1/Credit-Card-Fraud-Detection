"""
Model service for loading and running inference.

Handles model loading, caching, preprocessing, and prediction.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

import joblib
import numpy as np

from api.config import get_settings
from api.monitoring import (
    MODEL_LAST_USED,
    record_prediction,
    set_model_info,
    set_model_loaded,
)

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML model loading and inference."""

    def __init__(self):
        """Initialize the model service."""
        self.settings = get_settings()
        self.model = None
        self.model_type: Optional[str] = None
        self.loaded_at: Optional[datetime] = None
        self.load_time_seconds: float = 0.0
        self._metrics: Optional[dict] = None

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the ML model from disk.

        Args:
            model_path: Optional path to model file. Uses config default if not provided.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        path = Path(model_path or self.settings.model_path)

        if not path.exists():
            logger.error(f"Model file not found: {path}")
            set_model_loaded(False)
            return False

        try:
            start_time = time.time()
            self.model = joblib.load(path)
            self.load_time_seconds = time.time() - start_time
            self.loaded_at = datetime.utcnow()

            # Determine model type
            model_class = type(self.model).__name__
            if "XGB" in model_class:
                self.model_type = "XGBoost"
            elif "RandomForest" in model_class:
                self.model_type = "RandomForest"
            elif "LogisticRegression" in model_class:
                self.model_type = "LogisticRegression"
            else:
                self.model_type = model_class

            # Update metrics
            set_model_loaded(True, self.load_time_seconds)
            set_model_info(
                name=self.settings.model_name,
                version=self.settings.model_version,
                model_type=self.model_type
            )

            logger.info(
                f"Model loaded successfully: {self.model_type} "
                f"from {path} in {self.load_time_seconds:.3f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            set_model_loaded(False)
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"error": "Model not loaded"}

        return {
            "model_name": self.settings.model_name,
            "model_version": self.settings.model_version,
            "model_type": self.model_type,
            "features": self.settings.feature_columns,
            "training_date": None,  # Could be loaded from metadata
            "metrics": self._metrics,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "load_time_seconds": self.load_time_seconds,
        }

    def preprocess_input(self, features: dict[str, Any]) -> np.ndarray:
        """Preprocess input features for model prediction.

        Args:
            features: Dictionary of feature names to values.

        Returns:
            Numpy array ready for model prediction.
        """
        # Extract features in the correct order
        feature_values = []
        for col in self.settings.feature_columns:
            if col in features:
                feature_values.append(features[col])
            else:
                # Handle missing features with default value
                feature_values.append(0.0)

        return np.array([feature_values])

    def predict(self, features: dict[str, Any]) -> dict:
        """Make a single prediction.

        Args:
            features: Dictionary of feature names to values.

        Returns:
            Dictionary with prediction results.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        # Preprocess
        X = self.preprocess_input(features)

        # Predict
        prediction = int(self.model.predict(X)[0])

        # Get probability
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            fraud_probability = float(probabilities[1])
        else:
            # For models without predict_proba
            fraud_probability = float(prediction)

        # Calculate confidence
        confidence = max(fraud_probability, 1 - fraud_probability)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Generate transaction ID
        transaction_id = f"txn_{uuid.uuid4().hex[:12]}"

        # Record metrics
        result_label = "fraud" if prediction == 1 else "legitimate"
        record_prediction(result_label, fraud_probability, processing_time_ms / 1000)
        MODEL_LAST_USED.set(time.time())

        return {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "prediction_label": result_label,
            "fraud_probability": fraud_probability,
            "confidence": confidence,
            "model_version": self.settings.model_version,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.utcnow(),
        }

    def predict_batch(self, transactions: list[dict[str, Any]]) -> list[dict]:
        """Make batch predictions.

        Args:
            transactions: List of feature dictionaries.

        Returns:
            List of prediction results.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        results = []

        # Preprocess all inputs
        X = np.vstack([self.preprocess_input(t) for t in transactions])

        # Batch predict
        predictions = self.model.predict(X)

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
        else:
            probabilities = np.column_stack([1 - predictions, predictions])

        total_time = time.time() - start_time
        per_transaction_time = (total_time / len(transactions)) * 1000

        for i, (pred, probs, features) in enumerate(zip(predictions, probabilities, transactions)):
            prediction = int(pred)
            fraud_probability = float(probs[1])
            confidence = max(fraud_probability, 1 - fraud_probability)
            result_label = "fraud" if prediction == 1 else "legitimate"
            transaction_id = f"txn_{uuid.uuid4().hex[:12]}"

            # Record individual metrics
            record_prediction(result_label, fraud_probability, per_transaction_time / 1000)

            results.append({
                "transaction_id": transaction_id,
                "prediction": prediction,
                "prediction_label": result_label,
                "fraud_probability": fraud_probability,
                "confidence": confidence,
                "model_version": self.settings.model_version,
                "processing_time_ms": per_transaction_time,
                "timestamp": datetime.utcnow(),
            })

        MODEL_LAST_USED.set(time.time())
        return results

    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """Hot-reload the model from disk.

        Args:
            model_path: Optional new path to model file.

        Returns:
            True if reload successful, False otherwise.
        """
        logger.info("Hot-reloading model...")
        old_model = self.model
        old_model_type = self.model_type

        success = self.load_model(model_path)

        if not success:
            # Restore old model on failure
            self.model = old_model
            self.model_type = old_model_type
            logger.warning("Model reload failed, restored previous model")

        return success


# Global model service instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def initialize_model_service() -> bool:
    """Initialize and load the model service."""
    service = get_model_service()
    return service.load_model()
