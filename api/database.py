"""
Database management for prediction logging.

Uses SQLite with async support for storing predictions
and enabling drift analysis.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from api.config import get_settings
from api.monitoring import DB_CONNECTION_ERRORS, DB_PREDICTIONS_STORED

logger = logging.getLogger(__name__)


class PredictionDatabase:
    """Async SQLite database for storing predictions."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database with path."""
        settings = get_settings()
        self.db_path = db_path or settings.database_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Establish database connection and create tables."""
        try:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
            await self._create_tables()
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            DB_CONNECTION_ERRORS.inc()
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def disconnect(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    async def _create_tables(self):
        """Create required database tables."""
        await self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                prediction INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL,
                features TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
            CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);
            CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);

            CREATE TABLE IF NOT EXISTS drift_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                overall_score REAL NOT NULL,
                feature_scores TEXT NOT NULL,
                prediction_drift REAL NOT NULL,
                drift_detected INTEGER NOT NULL,
                alert_level TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_drift_checks_created_at ON drift_checks(created_at);

            CREATE TABLE IF NOT EXISTS model_loads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_path TEXT NOT NULL,
                load_time_seconds REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await self._connection.commit()

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        try:
            yield
            await self._connection.commit()
        except Exception as e:
            await self._connection.rollback()
            DB_CONNECTION_ERRORS.inc()
            raise

    async def store_prediction(
        self,
        transaction_id: str,
        prediction: int,
        fraud_probability: float,
        confidence: float,
        model_version: str,
        features: dict[str, Any],
        processing_time_ms: float
    ):
        """Store a prediction in the database."""
        async with self.transaction():
            await self._connection.execute(
                """
                INSERT INTO predictions (
                    transaction_id, prediction, fraud_probability, confidence,
                    model_version, features, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    prediction,
                    fraud_probability,
                    confidence,
                    model_version,
                    json.dumps(features),
                    processing_time_ms
                )
            )
            DB_PREDICTIONS_STORED.inc()

    async def get_recent_predictions(
        self,
        limit: int = 100,
        prediction_filter: Optional[int] = None
    ) -> list[dict]:
        """Get recent predictions from the database."""
        query = "SELECT * FROM predictions"
        params = []

        if prediction_filter is not None:
            query += " WHERE prediction = ?"
            params.append(prediction_filter)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_prediction_stats(
        self,
        hours: int = 24
    ) -> dict:
        """Get prediction statistics for the last N hours."""
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as fraud_count,
                SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as legitimate_count,
                AVG(fraud_probability) as avg_probability,
                AVG(processing_time_ms) as avg_processing_time
            FROM predictions
            WHERE created_at >= datetime('now', ?)
        """
        cursor = await self._connection.execute(query, (f"-{hours} hours",))
        row = await cursor.fetchone()
        return dict(row) if row else {}

    async def get_feature_distribution(
        self,
        feature_name: str,
        hours: int = 24,
        limit: int = 10000
    ) -> list[float]:
        """Get feature values for drift analysis."""
        query = """
            SELECT features FROM predictions
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            LIMIT ?
        """
        cursor = await self._connection.execute(query, (f"-{hours} hours", limit))
        rows = await cursor.fetchall()

        values = []
        for row in rows:
            features = json.loads(row["features"])
            if feature_name in features:
                values.append(features[feature_name])

        return values

    async def get_probability_distribution(
        self,
        hours: int = 24,
        limit: int = 10000
    ) -> list[float]:
        """Get fraud probability values for drift analysis."""
        query = """
            SELECT fraud_probability FROM predictions
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            LIMIT ?
        """
        cursor = await self._connection.execute(query, (f"-{hours} hours", limit))
        rows = await cursor.fetchall()
        return [row["fraud_probability"] for row in rows]

    async def store_drift_check(
        self,
        overall_score: float,
        feature_scores: dict[str, float],
        prediction_drift: float,
        drift_detected: bool,
        alert_level: str
    ):
        """Store drift check results."""
        async with self.transaction():
            await self._connection.execute(
                """
                INSERT INTO drift_checks (
                    overall_score, feature_scores, prediction_drift,
                    drift_detected, alert_level
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    overall_score,
                    json.dumps(feature_scores),
                    prediction_drift,
                    1 if drift_detected else 0,
                    alert_level
                )
            )

    async def get_latest_drift_check(self) -> Optional[dict]:
        """Get the most recent drift check result."""
        query = """
            SELECT * FROM drift_checks
            ORDER BY created_at DESC
            LIMIT 1
        """
        cursor = await self._connection.execute(query)
        row = await cursor.fetchone()
        if row:
            result = dict(row)
            result["feature_scores"] = json.loads(result["feature_scores"])
            return result
        return None

    async def store_model_load(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        load_time_seconds: float
    ):
        """Store model load event."""
        async with self.transaction():
            await self._connection.execute(
                """
                INSERT INTO model_loads (
                    model_name, model_version, model_path, load_time_seconds
                ) VALUES (?, ?, ?, ?)
                """,
                (model_name, model_version, model_path, load_time_seconds)
            )

    async def is_connected(self) -> bool:
        """Check if database connection is active."""
        if not self._connection:
            return False
        try:
            await self._connection.execute("SELECT 1")
            return True
        except Exception:
            return False


# Global database instance
_db: Optional[PredictionDatabase] = None


async def get_database() -> PredictionDatabase:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = PredictionDatabase()
        await _db.connect()
    return _db


async def close_database():
    """Close the global database connection."""
    global _db
    if _db is not None:
        await _db.disconnect()
        _db = None
