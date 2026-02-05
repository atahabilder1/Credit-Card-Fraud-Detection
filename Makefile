# Credit Card Fraud Detection - Makefile
# ======================================
# Common commands for development and deployment

.PHONY: help install dev prod test lint format clean train baseline simulate load-test docker-build docker-up docker-down logs

# Default target
help:
	@echo "Credit Card Fraud Detection - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development stack with Docker Compose"
	@echo "  make run          - Run API locally (without Docker)"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with Black"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production stack with Docker Compose"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start all Docker services"
	@echo "  make docker-down  - Stop all Docker services"
	@echo ""
	@echo "ML Operations:"
	@echo "  make train        - Train model and register to MLflow"
	@echo "  make baseline     - Generate drift detection baseline"
	@echo "  make simulate     - Run traffic simulator"
	@echo ""
	@echo "Testing:"
	@echo "  make load-test    - Run load tests with Locust"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean up cache and build files"
	@echo "  make logs         - View Docker logs"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Development
dev:
	docker-compose -f docker-compose.dev.yml up --build

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	pytest tests/ -v --cov=api --cov=src --cov=drift_detection --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

test-api:
	pytest tests/test_api.py -v

test-model:
	pytest tests/test_model_service.py -v

test-drift:
	pytest tests/test_drift.py -v

# Code Quality
lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

typecheck:
	mypy api/ src/ drift_detection/ --ignore-missing-imports

# Production
prod:
	docker-compose up -d

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-restart:
	docker-compose down
	docker-compose up -d

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

# ML Operations
train:
	python scripts/train_and_register.py --register --promote

train-quick:
	python scripts/train_and_register.py --quick

baseline:
	python scripts/generate_baseline.py

simulate:
	python -m scripts.traffic_simulator

# Load Testing
load-test:
	locust -f tests/load_tests/locustfile.py --host=http://localhost:8000

load-test-headless:
	locust -f tests/load_tests/locustfile.py --host=http://localhost:8000 \
		--headless -u 10 -r 2 -t 60s

# Cleaning
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

clean-docker:
	docker-compose down -v --rmi local
	docker system prune -f

# Database
db-reset:
	rm -f predictions.db
	@echo "Database reset complete"

# Health checks
health:
	curl -s http://localhost:8000/health | python -m json.tool

ready:
	curl -s http://localhost:8000/health/ready | python -m json.tool

stats:
	curl -s http://localhost:8000/stats | python -m json.tool

# Quick prediction test
predict-test:
	curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}' \
		| python -m json.tool
