"""
Pydantic schemas for request/response validation.

Defines all data models used by the API endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Input schema for a single transaction prediction."""

    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., ge=0, description="Transaction amount in dollars")
    Time: Optional[float] = Field(None, ge=0, description="Seconds elapsed since first transaction (optional)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                    "Amount": 149.62,
                    "Time": 0
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    transaction_id: str = Field(..., description="Unique identifier for this prediction")
    prediction: int = Field(..., description="Prediction result: 0=legitimate, 1=fraud")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    model_version: str = Field(..., description="Version of the model used")
    processing_time_ms: float = Field(..., description="Time taken for prediction in milliseconds")
    timestamp: datetime = Field(..., description="Timestamp of prediction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "txn_abc123",
                    "prediction": 0,
                    "prediction_label": "legitimate",
                    "fraud_probability": 0.0234,
                    "confidence": 0.9766,
                    "model_version": "1.0.0",
                    "processing_time_ms": 2.45,
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            ]
        }
    }


class BatchTransactionInput(BaseModel):
    """Input schema for batch prediction requests."""

    transactions: list[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to predict (max 1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    total_transactions: int = Field(..., description="Total number of transactions processed")
    fraud_count: int = Field(..., description="Number of transactions flagged as fraud")
    legitimate_count: int = Field(..., description="Number of legitimate transactions")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    average_processing_time_ms: float = Field(..., description="Average time per transaction")


class HealthResponse(BaseModel):
    """Response schema for health check endpoints."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server time")


class ReadinessResponse(BaseModel):
    """Response schema for readiness check."""

    ready: bool = Field(..., description="Whether the service is ready to accept requests")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    database_connected: bool = Field(..., description="Whether database is connected")
    details: Optional[dict] = Field(None, description="Additional details")


class ModelInfo(BaseModel):
    """Response schema for model information."""

    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    model_type: str = Field(..., description="Type of ML model (e.g., XGBoost)")
    features: list[str] = Field(..., description="List of input features")
    training_date: Optional[str] = Field(None, description="When the model was trained")
    metrics: Optional[dict] = Field(None, description="Model performance metrics")
    loaded_at: datetime = Field(..., description="When the model was loaded")


class DriftStatus(BaseModel):
    """Response schema for drift detection status."""

    drift_detected: bool = Field(..., description="Whether drift has been detected")
    overall_drift_score: float = Field(..., description="Overall drift score (0-1)")
    feature_drift_scores: dict[str, float] = Field(..., description="Per-feature drift scores")
    prediction_drift_score: float = Field(..., description="Drift in prediction distribution")
    last_checked: datetime = Field(..., description="When drift was last checked")
    baseline_date: Optional[str] = Field(None, description="When baseline was established")
    alert_level: str = Field(..., description="Alert level: none, warning, critical")


class APIInfo(BaseModel):
    """Response schema for API information."""

    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    developer: dict = Field(..., description="Developer information")
    endpoints: dict = Field(..., description="Available endpoints")
    documentation_url: str = Field(..., description="URL to API documentation")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="When the error occurred")


# Example data for Swagger UI - Fraudulent transaction
FRAUD_EXAMPLE = {
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
    "Amount": 0.0,
    "Time": 406
}

# Example data for Swagger UI - Legitimate transaction
LEGITIMATE_EXAMPLE = {
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
    "Amount": 149.62,
    "Time": 0
}
