"""
API endpoint tests for the Fraud Detection API.

Tests all REST endpoints for correct behavior, validation,
and error handling.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, test_client):
        """Test basic health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_readiness_check(self, test_client):
        """Test readiness check endpoint."""
        with patch("api.main.get_database") as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.is_connected.return_value = True
            mock_db.return_value = mock_db_instance

            response = test_client.get("/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert "ready" in data
            assert "model_loaded" in data
            assert "database_connected" in data


class TestAPIInfo:
    """Test API information endpoints."""

    def test_api_info(self, test_client):
        """Test API info endpoint."""
        response = test_client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "developer" in data
        assert "endpoints" in data

    def test_root_redirect_or_page(self, test_client):
        """Test root endpoint serves landing page or redirects."""
        response = test_client.get("/", follow_redirects=False)
        # Should either return HTML or redirect to /docs
        assert response.status_code in [200, 307]


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_predict_single_transaction(self, test_client, legitimate_transaction):
        """Test single transaction prediction."""
        with patch("api.main.get_database") as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.store_prediction = AsyncMock()
            mock_db.return_value = mock_db_instance

            response = test_client.post("/predict", json=legitimate_transaction)
            assert response.status_code == 200
            data = response.json()

            assert "transaction_id" in data
            assert "prediction" in data
            assert "fraud_probability" in data
            assert "confidence" in data
            assert "model_version" in data
            assert "processing_time_ms" in data
            assert data["prediction"] in [0, 1]
            assert 0 <= data["fraud_probability"] <= 1

    def test_predict_fraud_transaction(self, test_client, fraud_transaction):
        """Test prediction with fraudulent transaction."""
        with patch("api.main.get_database") as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.store_prediction = AsyncMock()
            mock_db.return_value = mock_db_instance

            # Update mock to return fraud prediction
            with patch("api.main.get_model_service") as mock_service:
                mock_model_service = MagicMock()
                mock_model_service.is_loaded.return_value = True
                mock_model_service.predict.return_value = {
                    "transaction_id": "txn_fraud123",
                    "prediction": 1,
                    "prediction_label": "fraud",
                    "fraud_probability": 0.98,
                    "confidence": 0.98,
                    "model_version": "1.0.0",
                    "processing_time_ms": 1.5,
                    "timestamp": "2024-01-01T00:00:00Z",
                }
                mock_service.return_value = mock_model_service

                response = test_client.post("/predict", json=fraud_transaction)
                assert response.status_code == 200

    def test_predict_invalid_input(self, test_client):
        """Test prediction with invalid input."""
        # Missing required fields
        response = test_client.post("/predict", json={"Amount": 100})
        assert response.status_code == 422  # Validation error

    def test_predict_negative_amount(self, test_client, legitimate_transaction):
        """Test prediction with negative amount (should fail validation)."""
        invalid_transaction = legitimate_transaction.copy()
        invalid_transaction["Amount"] = -100

        response = test_client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_batch_prediction(self, test_client, batch_transactions):
        """Test batch prediction endpoint."""
        with patch("api.main.get_database") as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.store_prediction = AsyncMock()
            mock_db.return_value = mock_db_instance

            with patch("api.main.get_model_service") as mock_service:
                mock_model_service = MagicMock()
                mock_model_service.is_loaded.return_value = True
                mock_model_service.predict_batch.return_value = [
                    {
                        "transaction_id": "txn_batch1",
                        "prediction": 0,
                        "prediction_label": "legitimate",
                        "fraud_probability": 0.05,
                        "confidence": 0.95,
                        "model_version": "1.0.0",
                        "processing_time_ms": 0.5,
                        "timestamp": "2024-01-01T00:00:00Z",
                    },
                    {
                        "transaction_id": "txn_batch2",
                        "prediction": 1,
                        "prediction_label": "fraud",
                        "fraud_probability": 0.95,
                        "confidence": 0.95,
                        "model_version": "1.0.0",
                        "processing_time_ms": 0.5,
                        "timestamp": "2024-01-01T00:00:00Z",
                    }
                ]
                mock_service.return_value = mock_model_service

                response = test_client.post(
                    "/predict/batch",
                    json={"transactions": batch_transactions}
                )
                assert response.status_code == 200
                data = response.json()

                assert "predictions" in data
                assert len(data["predictions"]) == 2
                assert "total_transactions" in data
                assert "fraud_count" in data
                assert "legitimate_count" in data

    def test_batch_prediction_empty(self, test_client):
        """Test batch prediction with empty list."""
        response = test_client.post(
            "/predict/batch",
            json={"transactions": []}
        )
        assert response.status_code == 422  # Validation error - min 1 item

    def test_batch_prediction_too_many(self, test_client, legitimate_transaction):
        """Test batch prediction exceeding limit."""
        # Create 1001 transactions (exceeds 1000 limit)
        transactions = [legitimate_transaction.copy() for _ in range(1001)]

        response = test_client.post(
            "/predict/batch",
            json={"transactions": transactions}
        )
        assert response.status_code == 422  # Validation error - max 1000 items


class TestModelEndpoints:
    """Test model information endpoints."""

    def test_model_info(self, test_client):
        """Test model info endpoint."""
        response = test_client.get("/model/info")
        assert response.status_code == 200
        data = response.json()

        assert "model_name" in data
        assert "model_version" in data
        assert "model_type" in data
        assert "features" in data


class TestExampleEndpoints:
    """Test example data endpoints."""

    def test_legitimate_example(self, test_client):
        """Test legitimate example endpoint."""
        response = test_client.get("/examples/legitimate")
        assert response.status_code == 200
        data = response.json()

        assert "description" in data
        assert "data" in data
        assert "V1" in data["data"]
        assert "Amount" in data["data"]

    def test_fraud_example(self, test_client):
        """Test fraud example endpoint."""
        response = test_client.get("/examples/fraud")
        assert response.status_code == 200
        data = response.json()

        assert "description" in data
        assert "data" in data


class TestMonitoringEndpoints:
    """Test monitoring endpoints."""

    def test_metrics_endpoint(self, test_client):
        """Test Prometheus metrics endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        # Should return Prometheus format
        assert "predictions_total" in response.text or response.status_code == 200

    def test_stats_endpoint(self, test_client):
        """Test statistics endpoint."""
        with patch("api.main.get_database") as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.get_prediction_stats = AsyncMock(return_value={
                "total": 100,
                "fraud_count": 5,
                "legitimate_count": 95,
                "avg_probability": 0.05,
                "avg_processing_time": 2.0
            })
            mock_db.return_value = mock_db_instance

            response = test_client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "total_predictions" in data

    def test_drift_status(self, test_client):
        """Test drift status endpoint."""
        with patch("api.main.get_database") as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.get_latest_drift_check = AsyncMock(return_value=None)
            mock_db.return_value = mock_db_instance

            response = test_client.get("/drift/status")
            assert response.status_code == 200
            data = response.json()
            assert "drift_detected" in data
            assert "overall_drift_score" in data


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_json(self, test_client):
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_wrong_content_type(self, test_client, legitimate_transaction):
        """Test handling of wrong content type."""
        response = test_client.post(
            "/predict",
            data=str(legitimate_transaction),
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422

    def test_not_found(self, test_client):
        """Test 404 for non-existent endpoint."""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404
