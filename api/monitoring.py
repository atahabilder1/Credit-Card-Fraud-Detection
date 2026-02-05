"""
Prometheus metrics definitions for the Fraud Detection API.

Defines counters, histograms, and gauges for monitoring API performance,
predictions, and model health.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# API Request Metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# Prediction Metrics
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["result"]  # fraud, legitimate
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken for model prediction in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of fraud probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

BATCH_SIZE = Histogram(
    "batch_prediction_size",
    "Size of batch prediction requests",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# Model Metrics
MODEL_INFO = Info(
    "model",
    "Information about the loaded model"
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time taken to load the model"
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is currently loaded (1=yes, 0=no)"
)

MODEL_LAST_USED = Gauge(
    "model_last_used_timestamp",
    "Unix timestamp of last model usage"
)

# Drift Metrics
DRIFT_SCORE = Gauge(
    "drift_score",
    "Current drift score",
    ["type"]  # overall, feature_*, prediction
)

DRIFT_ALERT = Gauge(
    "drift_alert_active",
    "Whether a drift alert is currently active (1=yes, 0=no)",
    ["severity"]  # warning, critical
)

# Database Metrics
DB_PREDICTIONS_STORED = Counter(
    "db_predictions_stored_total",
    "Total number of predictions stored in database"
)

DB_CONNECTION_ERRORS = Counter(
    "db_connection_errors_total",
    "Total number of database connection errors"
)

# Error Metrics
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"]
)


def record_prediction(result: str, probability: float, latency_seconds: float):
    """Record metrics for a single prediction."""
    PREDICTIONS_TOTAL.labels(result=result).inc()
    PREDICTION_PROBABILITY.observe(probability)
    PREDICTION_LATENCY.observe(latency_seconds)


def record_request(endpoint: str, method: str, status_code: int, latency_seconds: float):
    """Record metrics for an API request."""
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(latency_seconds)


def set_model_info(name: str, version: str, model_type: str):
    """Set model information gauge."""
    MODEL_INFO.info({
        "name": name,
        "version": version,
        "type": model_type
    })


def set_model_loaded(loaded: bool, load_time: float = 0.0):
    """Set model loaded status."""
    MODEL_LOADED.set(1 if loaded else 0)
    if load_time > 0:
        MODEL_LOAD_TIME.set(load_time)


def update_drift_scores(overall: float, feature_scores: dict[str, float], prediction_drift: float):
    """Update drift score metrics."""
    DRIFT_SCORE.labels(type="overall").set(overall)
    DRIFT_SCORE.labels(type="prediction").set(prediction_drift)
    for feature, score in feature_scores.items():
        DRIFT_SCORE.labels(type=f"feature_{feature}").set(score)


def set_drift_alert(severity: str, active: bool):
    """Set drift alert status."""
    DRIFT_ALERT.labels(severity=severity).set(1 if active else 0)
