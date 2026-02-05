"""
FastAPI application for Credit Card Fraud Detection.

Production-ready API with real-time prediction, monitoring,
and model management capabilities.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.config import get_settings
from api.database import close_database, get_database
from api.model_service import get_model_service, initialize_model_service
from api.monitoring import BATCH_SIZE, PREDICTION_ERRORS, record_request
from api.schemas import (
    APIInfo,
    BatchPredictionResponse,
    BatchTransactionInput,
    DriftStatus,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    ReadinessResponse,
    TransactionInput,
    FRAUD_EXAMPLE,
    LEGITIMATE_EXAMPLE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting Credit Card Fraud Detection API...")

    # Initialize model
    if not initialize_model_service():
        logger.warning("Model not loaded - API will start but predictions unavailable")

    # Initialize database
    try:
        await get_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    logger.info(f"API ready at http://{settings.api_host}:{settings.api_port}")

    yield

    # Shutdown
    logger.info("Shutting down API...")
    await close_database()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## Credit Card Fraud Detection API

Real-time machine learning API for detecting fraudulent credit card transactions.

### Features
- **Real-time Predictions**: Single and batch transaction classification
- **High Accuracy**: XGBoost model with 100% fraud recall
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Model Management**: MLflow experiment tracking and versioning

### Developer
- **Name**: Anik Tahabilder
- **Email**: tahabilderanik@gmail.com
- **LinkedIn**: [linkedin.com/in/tahabilder](https://www.linkedin.com/in/tahabilder/)

### Quick Start
Try the `/predict` endpoint with the example payload to test fraud detection.
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Add timing to all requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Record metrics
    record_request(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        latency_seconds=process_time
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ============== Root & Info Endpoints ==============

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the landing page or redirect to docs."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))

    # Fallback: redirect to API docs
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API</title>
        <meta http-equiv="refresh" content="0; url=/docs">
    </head>
    <body>
        <p>Redirecting to <a href="/docs">API Documentation</a>...</p>
    </body>
    </html>
    """)


@app.get("/api", response_model=APIInfo, tags=["Info"])
async def api_info():
    """Get API information and available endpoints."""
    return APIInfo(
        name=settings.app_name,
        version=settings.app_version,
        description="Real-time ML API for credit card fraud detection",
        developer={
            "name": settings.developer_name,
            "email": settings.developer_email,
            "linkedin": settings.developer_linkedin,
        },
        endpoints={
            "health": "/health",
            "readiness": "/health/ready",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "drift_status": "/drift/status",
            "documentation": "/docs",
        },
        documentation_url="/docs"
    )


# ============== Health Check Endpoints ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness probe - checks if the service is running."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.utcnow()
    )


@app.get("/health/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """Readiness probe - checks if the service can handle requests."""
    model_service = get_model_service()
    model_loaded = model_service.is_loaded()

    try:
        db = await get_database()
        db_connected = await db.is_connected()
    except Exception:
        db_connected = False

    ready = model_loaded and db_connected

    return ReadinessResponse(
        ready=ready,
        model_loaded=model_loaded,
        database_connected=db_connected,
        details={
            "model_type": model_service.model_type if model_loaded else None,
            "model_version": settings.model_version if model_loaded else None,
        }
    )


# ============== Prediction Endpoints ==============

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "examples": {
                        "legitimate": {
                            "summary": "Legitimate Transaction",
                            "value": {
                                "transaction_id": "txn_abc123def456",
                                "prediction": 0,
                                "prediction_label": "legitimate",
                                "fraud_probability": 0.0234,
                                "confidence": 0.9766,
                                "model_version": "1.0.0",
                                "processing_time_ms": 2.45,
                                "timestamp": "2024-01-15T10:30:00Z"
                            }
                        },
                        "fraud": {
                            "summary": "Fraudulent Transaction",
                            "value": {
                                "transaction_id": "txn_xyz789ghi012",
                                "prediction": 1,
                                "prediction_label": "fraud",
                                "fraud_probability": 0.9876,
                                "confidence": 0.9876,
                                "model_version": "1.0.0",
                                "processing_time_ms": 2.31,
                                "timestamp": "2024-01-15T10:30:05Z"
                            }
                        }
                    }
                }
            }
        },
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict(transaction: TransactionInput):
    """
    Predict if a transaction is fraudulent.

    **Input Features:**
    - V1-V28: PCA-transformed features from original transaction data
    - Amount: Transaction amount in dollars
    - Time: Seconds elapsed since first transaction (optional)

    **Output:**
    - prediction: 0 (legitimate) or 1 (fraud)
    - fraud_probability: Probability score between 0 and 1
    - confidence: Model confidence in the prediction

    **Example - Test a legitimate transaction:**
    ```json
    {
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
        "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
        "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
        "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
    }
    ```
    """
    model_service = get_model_service()

    if not model_service.is_loaded():
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up."
        )

    try:
        # Convert Pydantic model to dict, excluding Time if None
        features = transaction.model_dump(exclude_none=True)
        if "Time" in features:
            del features["Time"]  # Time is not used in prediction

        # Make prediction
        result = model_service.predict(features)

        # Store in database (async, don't wait)
        try:
            db = await get_database()
            await db.store_prediction(
                transaction_id=result["transaction_id"],
                prediction=result["prediction"],
                fraud_probability=result["fraud_probability"],
                confidence=result["confidence"],
                model_version=result["model_version"],
                features=features,
                processing_time_ms=result["processing_time_ms"]
            )
        except Exception as e:
            logger.warning(f"Failed to store prediction: {e}")

        return PredictionResponse(**result)

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="prediction_failed").inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"]
)
async def predict_batch(request: BatchTransactionInput):
    """
    Predict fraud for multiple transactions (batch processing).

    **Limits:**
    - Maximum 1000 transactions per request
    - Optimized for throughput with batch processing

    **Returns:**
    - Individual predictions for each transaction
    - Summary statistics (fraud count, processing time)
    """
    model_service = get_model_service()

    if not model_service.is_loaded():
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up."
        )

    try:
        start_time = time.time()

        # Convert to list of dicts
        transactions = [
            t.model_dump(exclude_none=True, exclude={"Time"})
            for t in request.transactions
        ]

        # Record batch size
        BATCH_SIZE.observe(len(transactions))

        # Make predictions
        results = model_service.predict_batch(transactions)

        # Store predictions in database
        try:
            db = await get_database()
            for result, features in zip(results, transactions):
                await db.store_prediction(
                    transaction_id=result["transaction_id"],
                    prediction=result["prediction"],
                    fraud_probability=result["fraud_probability"],
                    confidence=result["confidence"],
                    model_version=result["model_version"],
                    features=features,
                    processing_time_ms=result["processing_time_ms"]
                )
        except Exception as e:
            logger.warning(f"Failed to store batch predictions: {e}")

        total_time = (time.time() - start_time) * 1000
        fraud_count = sum(1 for r in results if r["prediction"] == 1)

        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_transactions=len(results),
            fraud_count=fraud_count,
            legitimate_count=len(results) - fraud_count,
            total_processing_time_ms=total_time,
            average_processing_time_ms=total_time / len(results)
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="batch_prediction_failed").inc()
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============== Model Endpoints ==============

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Get information about the currently loaded model."""
    model_service = get_model_service()

    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    info = model_service.get_model_info()
    return ModelInfo(
        model_name=info["model_name"],
        model_version=info["model_version"],
        model_type=info["model_type"],
        features=info["features"],
        training_date=info.get("training_date"),
        metrics=info.get("metrics"),
        loaded_at=datetime.fromisoformat(info["loaded_at"]) if info.get("loaded_at") else datetime.utcnow()
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Hot-reload the model from disk (admin endpoint)."""
    model_service = get_model_service()
    success = model_service.reload_model()

    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model"
        )


# ============== Monitoring Endpoints ==============

@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/drift/status", response_model=DriftStatus, tags=["Monitoring"])
async def drift_status():
    """Get current drift detection status."""
    try:
        db = await get_database()
        latest_check = await db.get_latest_drift_check()

        if latest_check:
            return DriftStatus(
                drift_detected=bool(latest_check["drift_detected"]),
                overall_drift_score=latest_check["overall_score"],
                feature_drift_scores=latest_check["feature_scores"],
                prediction_drift_score=latest_check["prediction_drift"],
                last_checked=latest_check["created_at"],
                baseline_date=None,
                alert_level=latest_check["alert_level"]
            )
        else:
            return DriftStatus(
                drift_detected=False,
                overall_drift_score=0.0,
                feature_drift_scores={},
                prediction_drift_score=0.0,
                last_checked=datetime.utcnow(),
                baseline_date=None,
                alert_level="none"
            )
    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve drift status"
        )


@app.get("/stats", tags=["Monitoring"])
async def prediction_stats():
    """Get prediction statistics for the last 24 hours."""
    try:
        db = await get_database()
        stats = await db.get_prediction_stats(hours=24)
        return {
            "period": "24h",
            "total_predictions": stats.get("total", 0),
            "fraud_detected": stats.get("fraud_count", 0),
            "legitimate": stats.get("legitimate_count", 0),
            "fraud_rate": (
                stats["fraud_count"] / stats["total"] * 100
                if stats.get("total", 0) > 0 else 0
            ),
            "avg_fraud_probability": stats.get("avg_probability", 0),
            "avg_processing_time_ms": stats.get("avg_processing_time", 0),
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {
            "period": "24h",
            "total_predictions": 0,
            "fraud_detected": 0,
            "legitimate": 0,
            "fraud_rate": 0,
            "avg_fraud_probability": 0,
            "avg_processing_time_ms": 0,
        }


# ============== Example Data Endpoints ==============

@app.get("/examples/legitimate", tags=["Examples"])
async def get_legitimate_example():
    """Get an example of a legitimate transaction for testing."""
    return {
        "description": "Example of a legitimate (non-fraudulent) transaction",
        "expected_result": "prediction=0 (legitimate)",
        "data": LEGITIMATE_EXAMPLE
    }


@app.get("/examples/fraud", tags=["Examples"])
async def get_fraud_example():
    """Get an example of a fraudulent transaction for testing."""
    return {
        "description": "Example of a fraudulent transaction",
        "expected_result": "prediction=1 (fraud)",
        "data": FRAUD_EXAMPLE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers
    )
