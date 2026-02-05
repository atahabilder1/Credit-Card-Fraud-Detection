# Credit Card Fraud Detection - Complete Technical Reference

**Author**: Anik Tahabilder
**Email**: tahabilderanik@gmail.com | gj9994@wayne.edu
**LinkedIn**: https://www.linkedin.com/in/tahabilder/

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Machine Learning Model](#3-machine-learning-model)
4. [API Reference](#4-api-reference)
5. [Monitoring & Observability](#5-monitoring--observability)
6. [Drift Detection](#6-drift-detection)
7. [MLflow Integration](#7-mlflow-integration)
8. [Testing Strategy](#8-testing-strategy)
9. [Deployment Guide](#9-deployment-guide)
10. [Maintenance & Operations](#10-maintenance--operations)
11. [Troubleshooting](#11-troubleshooting)
12. [Interview Reference](#12-interview-reference)

---

## 1. Project Overview

### 1.1 What This Project Does

This is a **production-ready MLOps system** for detecting fraudulent credit card transactions in real-time. It demonstrates end-to-end ML engineering skills including:

- **Real-time API serving** with FastAPI
- **Model monitoring** with Prometheus + Grafana
- **Experiment tracking** with MLflow
- **Automated drift detection** for model health
- **CI/CD pipelines** with GitHub Actions
- **Containerized deployment** with Docker Compose

### 1.2 Business Problem

Credit card fraud costs billions annually. The challenge:
- **Class imbalance**: Fraud is rare (~0.17% of transactions)
- **High stakes**: Missing fraud = direct financial loss
- **Real-time requirement**: Decisions needed in milliseconds
- **Drift over time**: Fraud patterns evolve

### 1.3 Solution Approach

1. **XGBoost classifier** trained on balanced data (SMOTE oversampling)
2. **FastAPI** for low-latency inference (<10ms)
3. **SQLite** for prediction logging (drift analysis)
4. **Prometheus metrics** for real-time monitoring
5. **Grafana dashboards** for visualization
6. **MLflow** for experiment tracking and model registry
7. **Statistical drift detection** (PSI, KS tests)

### 1.4 Key Metrics Achieved

| Metric | Value | Meaning |
|--------|-------|---------|
| **Recall** | 100.0% | Catches ALL fraud (zero missed) |
| **Precision** | 99.8% | Only 0.2% false alarms |
| **F1-Score** | 0.999 | Near-perfect balance |
| **ROC-AUC** | 1.000 | Perfect class separation |
| **Latency** | <10ms | Real-time capable |

---

## 2. Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│                         ┌─────────────┐                          │
│                         │   Client    │                          │
│                         │  (Browser)  │                          │
│                         └──────┬──────┘                          │
│                                │                                  │
│                                ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     FASTAPI (Port 8000)                  │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │    │
│  │  │ /predict│  │ /health │  │/metrics │  │ /docs   │   │    │
│  │  └────┬────┘  └─────────┘  └────┬────┘  └─────────┘   │    │
│  │       │                         │                       │    │
│  │       ▼                         │                       │    │
│  │  ┌─────────────┐               │                       │    │
│  │  │ModelService │               │                       │    │
│  │  │  (XGBoost)  │               │                       │    │
│  │  └──────┬──────┘               │                       │    │
│  │         │                       │                       │    │
│  │         ▼                       │                       │    │
│  │  ┌─────────────┐               │                       │    │
│  │  │   SQLite    │               │                       │    │
│  │  │ (Logging)   │               │                       │    │
│  │  └─────────────┘               │                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                   │                              │
│                                   │ Scrape /metrics              │
│                                   ▼                              │
│  ┌─────────────┐           ┌─────────────┐                      │
│  │ Prometheus  │◄──────────│   Grafana   │                      │
│  │  (9090)     │           │   (3000)    │                      │
│  └─────────────┘           └─────────────┘                      │
│                                                                   │
│  ┌─────────────┐           ┌─────────────┐                      │
│  │   MLflow    │           │  Traffic    │                      │
│  │   (5000)    │           │ Simulator   │                      │
│  └─────────────┘           └─────────────┘                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. **Transaction arrives** at `/predict` endpoint
2. **Pydantic validates** the 29 input features
3. **ModelService preprocesses** features (extract, order)
4. **XGBoost predicts** fraud probability
5. **Result logged** to SQLite database
6. **Prometheus metrics** updated
7. **Response returned** with prediction + confidence

### 2.3 Technology Stack

#### Tools & Frameworks Overview

| Tool/Framework | Category | What It Does | Why Chosen |
|----------------|----------|--------------|------------|
| **Python 3.10+** | Language | Core programming language | Industry standard for ML |
| **FastAPI** | API Framework | REST API with auto-docs | Async, fast, type hints |
| **Uvicorn** | ASGI Server | Production server | High performance |
| **Pydantic** | Validation | Request/response validation | Type safety, auto-docs |
| **XGBoost** | ML Model | Fraud classification | Best recall, fast inference |
| **scikit-learn** | ML Library | Preprocessing, metrics | Comprehensive, reliable |
| **imbalanced-learn** | ML Library | SMOTE oversampling | Handle class imbalance |
| **NumPy/Pandas** | Data Processing | Data manipulation | Industry standard |
| **joblib** | Serialization | Model save/load | Fast, efficient |
| **Prometheus** | Metrics | Metrics collection | Industry standard, pull-based |
| **Grafana** | Visualization | Dashboards, alerting | Rich UI, flexible |
| **MLflow** | MLOps | Experiment tracking | Model versioning, registry |
| **SQLite** | Database | Prediction logging | Simple, file-based |
| **aiosqlite** | Async DB | Non-blocking DB ops | Async support |
| **scipy** | Statistics | Drift detection (KS, PSI) | Statistical tests |
| **Docker** | Containerization | Package application | Portability |
| **Docker Compose** | Orchestration | Multi-container deployment | Easy orchestration |
| **GitHub Actions** | CI/CD | Automated pipelines | Native GitHub integration |
| **Pytest** | Testing | Unit/integration tests | Comprehensive, fixtures |
| **Locust** | Load Testing | Performance testing | Easy to use, scalable |
| **Ruff/Black** | Code Quality | Linting, formatting | Fast, consistent |

#### Component Mapping

| Component | Built With | Purpose |
|-----------|------------|---------|
| **API Server** | FastAPI + Uvicorn + Pydantic | Real-time prediction serving |
| **ML Pipeline** | XGBoost + scikit-learn + SMOTE | Model training and inference |
| **Prediction Logging** | SQLite + aiosqlite | Store predictions for drift analysis |
| **Metrics Export** | prometheus-client | Expose metrics for monitoring |
| **Dashboards** | Grafana + Prometheus | Visualize API health and predictions |
| **Experiment Tracking** | MLflow | Log training runs, compare models |
| **Model Registry** | MLflow | Version and deploy models |
| **Drift Detection** | scipy + custom algorithms | Detect data/model drift (PSI, KS tests) |
| **Traffic Simulator** | Python + requests | Generate realistic test traffic |
| **Landing Page** | HTML + CSS + JavaScript | Portfolio showcase with live demo |
| **CI Pipeline** | GitHub Actions | Run tests, lint, build on every push |
| **CD Pipeline** | GitHub Actions + SSH | Auto-deploy to production server |

---

## 3. Machine Learning Model

### 3.1 Model Selection

**Why XGBoost?**
- Handles class imbalance well with `scale_pos_weight`
- Fast inference (important for real-time)
- Feature importance for interpretability
- Robust to outliers
- Achieves 100% recall on fraud detection

**Alternatives Considered:**
- Random Forest: 99.9% recall (3 missed frauds)
- Logistic Regression: 95.2% recall (214 missed frauds)
- Neural Network: Overkill for tabular data, slower inference

### 3.2 Training Pipeline

```python
# Simplified training flow
from src.preprocess import DataPreprocessor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Load and preprocess
preprocessor = DataPreprocessor('data/creditcard_2023.csv')
X_train, X_test, y_train, y_test = preprocessor.split_data()

# 2. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 3. Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=1,  # Balanced after SMOTE
    eval_metric='logloss'
)
model.fit(X_train_balanced, y_train_balanced)

# 4. Save model
joblib.dump(model, 'models/xgboost_model.pkl')
```

### 3.3 Features

The model uses **29 features**:
- **V1-V28**: PCA-transformed features (anonymized for privacy)
- **Amount**: Transaction amount in dollars

**Feature Engineering:**
- Log transformation on Amount (optional)
- StandardScaler normalization
- No missing value handling needed (dataset is clean)

### 3.4 Hyperparameters

```python
{
    "n_estimators": 200,      # Number of trees
    "max_depth": 8,           # Tree depth (prevent overfitting)
    "learning_rate": 0.05,    # Step size shrinkage
    "min_child_weight": 1,    # Minimum sum of instance weight
    "subsample": 0.8,         # Row subsampling
    "colsample_bytree": 0.8,  # Feature subsampling
    "scale_pos_weight": 1,    # Class weight (balanced data)
    "eval_metric": "logloss"  # Binary cross-entropy
}
```

### 3.5 Model Performance

**Confusion Matrix (Test Set: 9,000 samples)**

```
                 PREDICTED
              Legit    Fraud
ACTUAL Legit   4492      8    ← Only 8 false positives
       Fraud      0   4500    ← Zero false negatives (100% recall)
```

**Key Insight**: The model catches **every single fraud** (100% recall) while only flagging 8 legitimate transactions incorrectly (99.8% precision).

### 3.6 Why 100% Recall Matters

In fraud detection, **recall is critical**:
- **Missed fraud (FN)** = Direct financial loss
- **False alarm (FP)** = Customer inconvenience (minor)

A model with 99% recall on 1M transactions would miss **~10,000 frauds**. Our 100% recall means **zero missed frauds**.

---

## 4. API Reference

### 4.1 Endpoints Overview

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | Landing page |
| `/docs` | GET | No | Swagger documentation |
| `/health` | GET | No | Liveness probe |
| `/health/ready` | GET | No | Readiness probe |
| `/predict` | POST | No | Single prediction |
| `/predict/batch` | POST | No | Batch prediction |
| `/model/info` | GET | No | Model metadata |
| `/metrics` | GET | No | Prometheus metrics |
| `/drift/status` | GET | No | Drift detection status |
| `/stats` | GET | No | Prediction statistics |

### 4.2 POST /predict

**Request:**
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

**Response:**
```json
{
  "transaction_id": "txn_abc123def456",
  "prediction": 0,
  "prediction_label": "legitimate",
  "fraud_probability": 0.0234,
  "confidence": 0.9766,
  "model_version": "1.0.0",
  "processing_time_ms": 2.45,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 4.3 POST /predict/batch

**Request:**
```json
{
  "transactions": [
    {"V1": -1.36, ..., "Amount": 149.62},
    {"V1": -2.31, ..., "Amount": 0.0}
  ]
}
```

**Response:**
```json
{
  "predictions": [...],
  "total_transactions": 2,
  "fraud_count": 1,
  "legitimate_count": 1,
  "total_processing_time_ms": 5.2,
  "average_processing_time_ms": 2.6
}
```

### 4.4 Error Handling

| Status | Meaning | Example |
|--------|---------|---------|
| 200 | Success | Prediction returned |
| 422 | Validation Error | Missing/invalid field |
| 500 | Server Error | Model inference failed |
| 503 | Unavailable | Model not loaded |

---

## 5. Monitoring & Observability

### 5.1 Prometheus Metrics

**Prediction Metrics:**
```
# Total predictions by result
predictions_total{result="fraud"} 150
predictions_total{result="legitimate"} 9850

# Prediction latency histogram
prediction_latency_seconds_bucket{le="0.01"} 9500
prediction_latency_seconds_bucket{le="0.05"} 9900

# Fraud probability distribution
prediction_probability_bucket{le="0.1"} 9500
prediction_probability_bucket{le="0.9"} 9850
```

**API Metrics:**
```
# Request count by endpoint
api_requests_total{endpoint="/predict", method="POST", status_code="200"} 10000

# Request latency
api_request_latency_seconds_bucket{endpoint="/predict", le="0.1"} 9900
```

**Model Metrics:**
```
model_loaded 1
model_load_time_seconds 0.45
```

### 5.2 Grafana Dashboard

**Row 1: API Health**
- Request rate (req/sec)
- Error rate (%)
- Latency p50/p95/p99

**Row 2: Predictions**
- Predictions over time (fraud vs legitimate)
- Fraud detection rate (%)
- Fraud probability histogram

**Row 3: Model Health**
- Model version info
- Inference latency trend
- Drift scores

### 5.3 Alerting

Configure in Grafana:
- **High Error Rate**: >5% 5xx responses
- **High Latency**: p95 > 100ms
- **Drift Alert**: PSI > 0.2
- **Model Down**: model_loaded = 0

---

## 6. Drift Detection

### 6.1 What is Drift?

**Data Drift**: Input feature distributions change over time
**Concept Drift**: Relationship between features and target changes
**Model Drift**: Model predictions shift from baseline

### 6.2 Detection Methods

**PSI (Population Stability Index)**
```python
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

Interpretation:
- PSI < 0.1: No significant change
- PSI 0.1-0.2: Moderate change (monitor)
- PSI > 0.2: Significant change (investigate/retrain)
```

**KS Test (Kolmogorov-Smirnov)**
- Compares cumulative distributions
- Higher KS statistic = more drift

### 6.3 Implementation

```python
from drift_detection.detector import DriftDetector
from drift_detection.baseline import BaselineComputer

# 1. Create baseline from training data
computer = BaselineComputer()
baseline = computer.compute_baseline(training_data)
computer.save_baseline('models/drift_baseline.json')

# 2. Detect drift in production
detector = DriftDetector(baseline)
result = detector.detect_drift(current_data)

if result.drift_detected:
    print(f"Alert: {result.alert_level}")
    for feature, score in result.feature_scores.items():
        if score > 0.2:
            print(f"  {feature}: PSI={score:.3f}")
```

### 6.4 Monitoring Schedule

- **Real-time**: Log all predictions to SQLite
- **Hourly**: Check prediction distribution drift
- **Daily**: Check feature distribution drift
- **Weekly**: Full drift report

---

## 7. MLflow Integration

### 7.1 Experiment Tracking

```python
from mlflow_tracking.experiment import ExperimentTracker

tracker = ExperimentTracker(experiment_name="fraud-detection")
tracker.start_run(run_name="xgboost_v2")

# Log parameters
tracker.log_params({
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05
})

# Log metrics
tracker.log_metrics({
    "accuracy": 0.999,
    "precision": 0.998,
    "recall": 1.0,
    "f1": 0.999,
    "roc_auc": 1.0
})

# Log model
tracker.log_model(model, registered_model_name="fraud_detector")
tracker.end_run()
```

### 7.2 Model Registry

**Stages:**
- `None` → New model versions
- `Staging` → Ready for testing
- `Production` → Serving live traffic
- `Archived` → Old versions

```python
from mlflow_tracking.registry import ModelRegistry

registry = ModelRegistry()

# Promote to production
registry.promote_to_production("fraud_detector", version="3")

# Load production model
model = registry.load_production_model("fraud_detector")
```

---

## 8. Testing Strategy

### 8.1 Test Pyramid

```
        /\
       /  \     E2E Tests (Docker Compose)
      /----\    Integration Tests (API)
     /      \   Unit Tests (Components)
    /--------\
```

### 8.2 Unit Tests

```bash
pytest tests/test_model_service.py -v  # Model loading, inference
pytest tests/test_drift.py -v          # Drift detection algorithms
```

### 8.3 Integration Tests

```bash
pytest tests/test_api.py -v            # API endpoints
```

### 8.4 Load Tests

```bash
# Locust load testing
locust -f tests/load_tests/locustfile.py --host=http://localhost:8000

# Headless mode
locust --headless -u 100 -r 10 -t 60s
```

### 8.5 Coverage

Target: **>80% coverage**
```bash
pytest --cov=api --cov=drift_detection --cov-report=html
```

---

## 9. Deployment Guide

### 9.1 Local Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run API with hot reload
uvicorn api.main:app --reload --port 8000
```

### 9.2 Docker Compose (Recommended)

```bash
# Development (with hot reload)
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose up -d
```

### 9.3 Production Deployment (Mac Mini)

**Pre-requisites:**
- Docker + Docker Compose installed
- SSH access configured
- GitHub secrets configured:
  - `DEPLOY_HOST`
  - `DEPLOY_USER`
  - `DEPLOY_SSH_KEY`
  - `DEPLOY_PATH`

**Deployment flow:**
1. Push to `main` branch
2. CI pipeline runs tests
3. CD pipeline builds Docker images
4. Images pushed to GHCR
5. SSH to Mac Mini
6. Pull new images
7. Restart services

### 9.4 Manual Deployment

```bash
# On production server
cd ~/fraud-detection
git pull origin main
docker-compose pull
docker-compose up -d
```

---

## 10. Maintenance & Operations

### 10.1 Daily Operations

- [ ] Check Grafana dashboards for anomalies
- [ ] Review error rates in Prometheus
- [ ] Check drift status at `/drift/status`

### 10.2 Weekly Operations

- [ ] Review MLflow experiments
- [ ] Analyze prediction distribution
- [ ] Check for feature drift

### 10.3 Monthly Operations

- [ ] Evaluate model retraining need
- [ ] Review and update documentation
- [ ] Security patches and updates

### 10.4 Model Retraining Triggers

1. **Drift detected**: PSI > 0.2 on key features
2. **Performance degradation**: Precision drops below 95%
3. **New fraud patterns**: Feedback from fraud team
4. **Scheduled**: Quarterly retraining

### 10.5 Retraining Process

```bash
# 1. Train new model
python scripts/train_and_register.py --register

# 2. Evaluate in staging
# Manual review in MLflow UI

# 3. Promote to production
python -c "
from mlflow_tracking.registry import ModelRegistry
registry = ModelRegistry()
registry.promote_to_production('fraud_detector', version='NEW_VERSION')
"

# 4. Restart API to load new model
docker-compose restart api
```

---

## 11. Troubleshooting

### 11.1 Common Issues

**API returns 503 (Model not loaded)**
```bash
# Check if model file exists
ls -la models/xgboost_model.pkl

# Check API logs
docker-compose logs api
```

**High latency (>100ms)**
```bash
# Check container resources
docker stats

# Check model load time
curl http://localhost:8000/model/info
```

**Drift alerts firing**
```bash
# Check drift status
curl http://localhost:8000/drift/status

# Generate new baseline if needed
python scripts/generate_baseline.py
```

### 11.2 Health Checks

```bash
# Liveness
curl http://localhost:8000/health

# Readiness
curl http://localhost:8000/health/ready

# Model info
curl http://localhost:8000/model/info
```

### 11.3 Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f prometheus
```

---

## 12. Interview Reference

### 12.1 Key Talking Points

1. **Problem**: Fraud detection with imbalanced data
2. **Solution**: XGBoost with SMOTE, achieving 100% recall
3. **Architecture**: FastAPI + Prometheus + Grafana + MLflow
4. **MLOps**: CI/CD, monitoring, drift detection
5. **Results**: Production-ready system with zero missed frauds

### 12.2 Technical Deep Dives

**Q: Why XGBoost over Deep Learning?**
- Tabular data works well with tree-based models
- Faster inference (<10ms vs 50ms+ for DL)
- More interpretable (feature importance)
- Less data needed

**Q: How do you handle class imbalance?**
- SMOTE oversampling during training
- Class weights in model
- Focus on recall metric (cost-sensitive)

**Q: How do you know when to retrain?**
- Drift detection (PSI, KS tests)
- Performance monitoring (precision/recall)
- Scheduled evaluation

**Q: What happens if the model fails?**
- Health checks detect issues
- Graceful degradation (503 response)
- Alerting via Grafana
- Rollback to previous model version

### 12.3 Code Highlights

**Model Service (api/model_service.py:90-120)**
- Singleton pattern for model loading
- Preprocessing pipeline
- Inference with probability

**Drift Detection (drift_detection/detector.py:50-100)**
- PSI calculation
- KS test implementation
- Alert thresholds

**Monitoring (api/monitoring.py)**
- Prometheus counters, histograms, gauges
- Custom metrics for ML

### 12.4 Demo Script

```bash
# 1. Start the stack
docker-compose up -d

# 2. Show landing page
open http://localhost:8000

# 3. Show API docs
open http://localhost:8000/docs

# 4. Make a test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1":-1.36,...,"Amount":149.62}'

# 5. Show Grafana dashboard
open http://localhost:3000

# 6. Show MLflow experiments
open http://localhost:5000

# 7. Check drift status
curl http://localhost:8000/drift/status
```

---

## Appendix

### A. Environment Variables

See `.env.example` for all configuration options.

### B. API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 422 | Validation error |
| 500 | Internal error |
| 503 | Service unavailable |

### C. Metrics Reference

See `api/monitoring.py` for all metric definitions.

---

*Last updated: February 2026*
