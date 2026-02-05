#!/usr/bin/env python3
"""
Generate drift detection baseline from training data.

This script computes baseline statistics from the training dataset
that will be used to detect drift in production.

Usage:
    python scripts/generate_baseline.py [--data-path DATA_PATH] [--output OUTPUT]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drift_detection.baseline import BaselineComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate drift detection baseline from training data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard_2023.csv",
        help="Path to the training data CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/drift_baseline.json",
        help="Output path for baseline JSON"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for large datasets (optional)"
    )
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Include prediction distribution (requires model)"
    )

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please ensure the dataset is available.")
        sys.exit(1)

    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} samples")

    # Sample if requested
    if args.sample_size and len(data) > args.sample_size:
        data = data.sample(n=args.sample_size, random_state=42)
        logger.info(f"Sampled to {len(data)} samples")

    # Feature columns
    feature_columns = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]

    # Verify columns exist
    available_columns = [c for c in feature_columns if c in data.columns]
    if len(available_columns) < len(feature_columns):
        missing = set(feature_columns) - set(available_columns)
        logger.warning(f"Missing columns: {missing}")

    # Compute prediction probabilities if requested
    prediction_probs = None
    if args.include_predictions:
        try:
            import joblib
            model_path = Path("models/xgboost_model.pkl")
            if model_path.exists():
                model = joblib.load(model_path)
                X = data[available_columns]
                prediction_probs = model.predict_proba(X)[:, 1]
                logger.info("Generated prediction probabilities")
            else:
                logger.warning("Model not found, skipping prediction distribution")
        except Exception as e:
            logger.warning(f"Could not generate predictions: {e}")

    # Compute baseline
    computer = BaselineComputer(feature_columns=available_columns)
    baseline = computer.compute_baseline(data, prediction_probs)

    # Save baseline
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    computer.save_baseline(str(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Data source: {data_path}")
    print(f"Samples analyzed: {baseline['metadata']['n_samples']}")
    print(f"Features tracked: {len(baseline['features'])}")
    print(f"Output saved to: {output_path}")
    print("=" * 60)

    # Show feature summary
    print("\nFeature Statistics Summary:")
    print("-" * 60)
    for feature, stats in list(baseline["features"].items())[:5]:
        print(f"  {feature}:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    if len(baseline["features"]) > 5:
        print(f"  ... and {len(baseline['features']) - 5} more features")


if __name__ == "__main__":
    main()
