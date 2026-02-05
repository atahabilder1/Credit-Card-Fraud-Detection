#!/usr/bin/env python3
"""
Train model and register to MLflow.

This script runs the full training pipeline, logs the experiment to MLflow,
and registers the best model to the model registry.

Usage:
    python scripts/train_and_register.py [--data-path DATA_PATH] [--model-type MODEL_TYPE]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import DataPreprocessor
from src.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    data_path: str,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
    quick_mode: bool = False
) -> tuple:
    """Train and evaluate a fraud detection model.

    Args:
        data_path: Path to the dataset.
        model_type: Type of model to train (xgboost, random_forest, logistic_regression).
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.
        use_smote: Whether to use SMOTE for class balancing.
        quick_mode: If True, use simplified hyperparameters.

    Returns:
        Tuple of (model, metrics, params).
    """
    logger.info(f"Loading data from {data_path}")

    # Load and preprocess data
    preprocessor = DataPreprocessor(
        filepath=data_path,
        test_size=test_size,
        random_state=random_state
    )
    preprocessor.load_data()
    preprocessor.clean_data()
    X_train, X_test, y_train, y_test = preprocessor.split_data()

    # Apply SMOTE if requested
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"Applied SMOTE: {len(X_train)} training samples")

    # Training parameters
    if model_type == "xgboost":
        if quick_mode:
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1,
                "random_state": random_state,
                "n_jobs": -1,
                "use_label_encoder": False,
                "eval_metric": "logloss"
            }
        else:
            params = {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.05,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 1,
                "random_state": random_state,
                "n_jobs": -1,
                "use_label_encoder": False,
                "eval_metric": "logloss"
            }

        from xgboost import XGBClassifier
        model = XGBClassifier(**params)

    elif model_type == "random_forest":
        if quick_mode:
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": random_state,
                "n_jobs": -1
            }
        else:
            params = {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": random_state,
                "n_jobs": -1
            }

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)

    elif model_type == "logistic_regression":
        params = {
            "max_iter": 1000,
            "random_state": random_state,
            "n_jobs": -1
        }

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    logger.info(f"Training {model_type} model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "test_samples": len(y_test),
        "train_samples": len(y_train),
        "fraud_test": int(y_test.sum()),
        "legit_test": int(len(y_test) - y_test.sum())
    }

    logger.info(f"Evaluation metrics: {metrics}")

    # Add preprocessing params
    params["test_size"] = test_size
    params["use_smote"] = use_smote
    params["model_type"] = model_type

    return model, metrics, params


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train fraud detection model and register to MLflow"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard_2023.csv",
        help="Path to the training data"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "logistic_regression"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick mode with simplified hyperparameters"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the model to MLflow"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the model to production"
    )

    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        logger.info("Please download the dataset first.")
        sys.exit(1)

    # Train and evaluate
    model, metrics, params = train_and_evaluate(
        data_path=args.data_path,
        model_type=args.model_type,
        quick_mode=args.quick
    )

    # Save model locally
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model_type}_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Log to MLflow if requested
    if args.register:
        try:
            from mlflow_tracking.experiment import log_training_run
            from mlflow_tracking.registry import ModelRegistry

            run_id = log_training_run(
                model=model,
                params=params,
                metrics=metrics,
                run_name=f"{args.model_type}_training",
                register_model=True,
                model_name="fraud_detector"
            )
            logger.info(f"Logged to MLflow, run_id: {run_id}")

            # Promote to production if requested
            if args.promote:
                registry = ModelRegistry()
                latest_version = registry.get_latest_version("fraud_detector")
                if latest_version:
                    registry.promote_to_production("fraud_detector", latest_version)
                    logger.info(f"Promoted model v{latest_version} to production")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
            logger.info("Model saved locally but not registered")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Model Path: {model_path}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
