"""
Tests for the model service module.

Tests model loading, inference, preprocessing, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest


class TestModelService:
    """Tests for ModelService class."""

    def test_model_service_initialization(self):
        """Test model service can be initialized."""
        from api.model_service import ModelService
        service = ModelService()

        assert service.model is None
        assert service.model_type is None
        assert service.loaded_at is None

    def test_is_loaded_false_initially(self):
        """Test is_loaded returns False when no model loaded."""
        from api.model_service import ModelService
        service = ModelService()

        assert service.is_loaded() is False

    def test_load_model_file_not_found(self):
        """Test load_model returns False when file doesn't exist."""
        from api.model_service import ModelService
        service = ModelService()

        result = service.load_model("/nonexistent/path/model.pkl")
        assert result is False
        assert service.is_loaded() is False

    def test_load_model_success(self, mock_model, temp_model_path):
        """Test successful model loading."""
        # Save mock model to temp file
        joblib.dump(mock_model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()

        with patch.object(service.settings, "model_path", temp_model_path):
            result = service.load_model(temp_model_path)

        assert result is True
        assert service.is_loaded() is True
        assert service.loaded_at is not None

    def test_preprocess_input(self, legitimate_transaction, feature_columns):
        """Test input preprocessing."""
        from api.model_service import ModelService
        service = ModelService()

        with patch.object(service.settings, "feature_columns", feature_columns):
            X = service.preprocess_input(legitimate_transaction)

        assert isinstance(X, np.ndarray)
        assert X.shape == (1, len(feature_columns))

    def test_preprocess_input_missing_feature(self, feature_columns):
        """Test preprocessing with missing feature uses default."""
        from api.model_service import ModelService
        service = ModelService()

        # Transaction with missing V1
        transaction = {"V2": 0.5, "Amount": 100.0}

        with patch.object(service.settings, "feature_columns", feature_columns):
            X = service.preprocess_input(transaction)

        assert X.shape == (1, len(feature_columns))
        # Missing features should be 0.0
        assert X[0, 0] == 0.0  # V1 is missing

    def test_predict_not_loaded(self, legitimate_transaction):
        """Test predict raises error when model not loaded."""
        from api.model_service import ModelService
        service = ModelService()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.predict(legitimate_transaction)

    def test_predict_success(self, mock_model, temp_model_path, legitimate_transaction):
        """Test successful prediction."""
        joblib.dump(mock_model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()
        service.load_model(temp_model_path)

        result = service.predict(legitimate_transaction)

        assert "transaction_id" in result
        assert "prediction" in result
        assert "fraud_probability" in result
        assert "confidence" in result
        assert "processing_time_ms" in result
        assert result["prediction"] in [0, 1]
        assert 0 <= result["fraud_probability"] <= 1

    def test_predict_batch(self, mock_model, temp_model_path, batch_transactions):
        """Test batch prediction."""
        # Configure mock for batch
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.95, 0.05],
            [0.1, 0.9]
        ])
        joblib.dump(mock_model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()
        service.load_model(temp_model_path)

        results = service.predict_batch(batch_transactions)

        assert len(results) == 2
        assert all("transaction_id" in r for r in results)
        assert all("prediction" in r for r in results)

    def test_get_model_info_not_loaded(self):
        """Test get_model_info when model not loaded."""
        from api.model_service import ModelService
        service = ModelService()

        info = service.get_model_info()
        assert "error" in info

    def test_get_model_info_loaded(self, mock_model, temp_model_path):
        """Test get_model_info when model is loaded."""
        joblib.dump(mock_model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()
        service.load_model(temp_model_path)

        info = service.get_model_info()

        assert "model_name" in info
        assert "model_version" in info
        assert "features" in info
        assert "load_time_seconds" in info

    def test_reload_model(self, mock_model, temp_model_path):
        """Test model hot-reload."""
        joblib.dump(mock_model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()
        service.load_model(temp_model_path)

        old_loaded_at = service.loaded_at

        # Reload
        result = service.reload_model(temp_model_path)

        assert result is True
        assert service.loaded_at >= old_loaded_at

    def test_reload_model_failure_restores_old(self, mock_model, temp_model_path):
        """Test reload failure restores previous model."""
        joblib.dump(mock_model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()
        service.load_model(temp_model_path)

        old_model = service.model

        # Try to reload from nonexistent path
        result = service.reload_model("/nonexistent/model.pkl")

        assert result is False
        assert service.model is old_model


class TestModelServiceSingleton:
    """Test model service singleton functions."""

    def test_get_model_service(self):
        """Test getting model service singleton."""
        from api.model_service import get_model_service

        service1 = get_model_service()
        service2 = get_model_service()

        # Should be same instance
        assert service1 is service2

    def test_initialize_model_service(self, mock_model, temp_model_path):
        """Test initializing model service."""
        joblib.dump(mock_model, temp_model_path)

        from api.model_service import initialize_model_service, get_model_service

        with patch.object(get_model_service().settings, "model_path", temp_model_path):
            # Reset model
            service = get_model_service()
            service.model = None

            result = initialize_model_service()

        assert result is True


class TestModelTypeDetection:
    """Test model type detection."""

    def test_detect_xgboost(self, temp_model_path):
        """Test XGBoost model type detection."""
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric="logloss")
            # Need to fit on dummy data
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 2, 100)
            model.fit(X, y)
            joblib.dump(model, temp_model_path)

            from api.model_service import ModelService
            service = ModelService()
            service.load_model(temp_model_path)

            assert service.model_type == "XGBoost"
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_detect_random_forest(self, temp_model_path):
        """Test RandomForest model type detection."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        joblib.dump(model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()
        service.load_model(temp_model_path)

        assert service.model_type == "RandomForest"

    def test_detect_logistic_regression(self, temp_model_path):
        """Test LogisticRegression model type detection."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        joblib.dump(model, temp_model_path)

        from api.model_service import ModelService
        service = ModelService()
        service.load_model(temp_model_path)

        assert service.model_type == "LogisticRegression"
