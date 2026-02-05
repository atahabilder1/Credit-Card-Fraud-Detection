"""
Pytest fixtures and configuration for the test suite.

Provides reusable fixtures for testing the API, model service,
and drift detection components.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Sample transaction data
@pytest.fixture
def legitimate_transaction():
    """Sample legitimate transaction data."""
    return {
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }


@pytest.fixture
def fraud_transaction():
    """Sample fraudulent transaction data."""
    return {
        "V1": -2.3122265423263,
        "V2": 1.95199201064158,
        "V3": -1.60985073229769,
        "V4": 3.9979055875468,
        "V5": -0.522187864667764,
        "V6": -1.42654531920595,
        "V7": -2.53738730624579,
        "V8": 1.39165724829804,
        "V9": -2.77008927719433,
        "V10": -2.77227214465915,
        "V11": 3.20203320709635,
        "V12": -2.89990738849473,
        "V13": -0.595221881324605,
        "V14": -4.28925378244217,
        "V15": 0.389724120274487,
        "V16": -1.14074717980657,
        "V17": -2.83005567450437,
        "V18": -0.0168224681808257,
        "V19": 0.416955705037907,
        "V20": 0.126910559061474,
        "V21": 0.517232370861764,
        "V22": -0.0350493686052974,
        "V23": -0.465211076182388,
        "V24": 0.320198198514526,
        "V25": 0.0445191674731724,
        "V26": 0.177839798284401,
        "V27": 0.261145002567677,
        "V28": -0.143275874698919,
        "Amount": 0.0
    }


@pytest.fixture
def batch_transactions(legitimate_transaction, fraud_transaction):
    """Batch of transactions for testing."""
    return [legitimate_transaction, fraud_transaction]


@pytest.fixture
def mock_model():
    """Mock ML model for testing without loading actual model."""
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.95, 0.05]])
    return model


@pytest.fixture
def temp_db_path():
    """Temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    # Cleanup
    try:
        os.unlink(f.name)
    except Exception:
        pass


@pytest.fixture
def temp_model_path():
    """Temporary model path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        yield f.name
    try:
        os.unlink(f.name)
    except Exception:
        pass


@pytest.fixture
def sample_baseline():
    """Sample drift detection baseline."""
    return {
        "metadata": {
            "n_samples": 10000,
            "created_at": "2024-01-01T00:00:00",
        },
        "features": {
            "V1": {
                "mean": 0.0,
                "std": 1.5,
                "min": -5.0,
                "max": 5.0,
                "median": 0.0,
                "q1": -1.0,
                "q3": 1.0,
                "skew": 0.0,
                "kurtosis": 0.0,
                "histogram": {
                    "bin_edges": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                    "counts": [100, 300, 800, 1500, 2500, 2500, 1500, 800, 300, 100],
                    "proportions": [0.01, 0.03, 0.08, 0.15, 0.25, 0.25, 0.15, 0.08, 0.03, 0.01]
                }
            },
            "Amount": {
                "mean": 88.0,
                "std": 250.0,
                "min": 0.0,
                "max": 25000.0,
                "median": 22.0,
                "q1": 5.0,
                "q3": 77.0,
                "skew": 16.0,
                "kurtosis": 350.0,
                "histogram": {
                    "bin_edges": [0, 50, 100, 200, 500, 1000, 2500, 5000, 10000, 25000],
                    "counts": [5000, 2000, 1500, 800, 400, 200, 70, 20, 10],
                    "proportions": [0.5, 0.2, 0.15, 0.08, 0.04, 0.02, 0.007, 0.002, 0.001]
                }
            }
        },
        "prediction_distribution": {
            "mean": 0.02,
            "std": 0.1,
            "median": 0.001,
            "histogram": {
                "bin_edges": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "counts": [9500, 200, 100, 50, 30, 20, 20, 20, 30, 30],
                "proportions": [0.95, 0.02, 0.01, 0.005, 0.003, 0.002, 0.002, 0.002, 0.003, 0.003]
                }
        }
    }


# FastAPI test client fixture
@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    # Avoid importing at module level to prevent issues
    from fastapi.testclient import TestClient

    # Patch model loading to use mock
    with patch("api.model_service.get_model_service") as mock_service:
        mock_model_service = MagicMock()
        mock_model_service.is_loaded.return_value = True
        mock_model_service.predict.return_value = {
            "transaction_id": "txn_test123",
            "prediction": 0,
            "prediction_label": "legitimate",
            "fraud_probability": 0.05,
            "confidence": 0.95,
            "model_version": "1.0.0",
            "processing_time_ms": 1.5,
            "timestamp": "2024-01-01T00:00:00Z",
        }
        mock_model_service.get_model_info.return_value = {
            "model_name": "fraud_detector",
            "model_version": "1.0.0",
            "model_type": "XGBoost",
            "features": ["V1", "V2", "Amount"],
            "loaded_at": "2024-01-01T00:00:00",
            "load_time_seconds": 0.5,
        }
        mock_service.return_value = mock_model_service

        # Import app after patching
        from api.main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def feature_columns():
    """List of feature columns."""
    return [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
