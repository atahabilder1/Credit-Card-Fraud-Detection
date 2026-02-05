"""
Configuration management for the Fraud Detection API.

Uses Pydantic Settings for environment variable loading with defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Credit Card Fraud Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Model
    model_path: str = "models/xgboost_model.pkl"
    model_name: str = "xgboost_fraud_detector"
    model_version: str = "1.0.0"

    # Database
    database_url: str = "sqlite+aiosqlite:///./predictions.db"
    database_path: str = "predictions.db"

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "fraud-detection"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8000

    # Drift Detection
    drift_baseline_path: str = "models/drift_baseline.json"
    drift_check_interval: int = 3600  # seconds
    drift_threshold_psi: float = 0.2
    drift_threshold_ks: float = 0.1

    # Feature configuration
    feature_columns: list[str] = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]

    # Developer info
    developer_name: str = "Anik Tahabilder"
    developer_email: str = "tahabilderanik@gmail.com"
    developer_linkedin: str = "https://www.linkedin.com/in/tahabilder/"

    @property
    def model_file_path(self) -> Path:
        """Get the model file path as a Path object."""
        return Path(self.model_path)

    @property
    def baseline_file_path(self) -> Path:
        """Get the drift baseline file path as a Path object."""
        return Path(self.drift_baseline_path)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
