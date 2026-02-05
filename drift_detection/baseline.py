"""
Baseline statistics computation for drift detection.

Computes and stores reference statistics from training data
that are used to detect drift in production.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaselineComputer:
    """Compute and manage baseline statistics for drift detection."""

    def __init__(self, feature_columns: Optional[list[str]] = None):
        """Initialize the baseline computer.

        Args:
            feature_columns: List of feature column names to track.
        """
        self.feature_columns = feature_columns or [
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]
        self.baseline: Optional[dict] = None

    def compute_baseline(
        self,
        data: pd.DataFrame,
        prediction_probabilities: Optional[np.ndarray] = None
    ) -> dict:
        """Compute baseline statistics from training data.

        Args:
            data: Training data DataFrame.
            prediction_probabilities: Optional array of prediction probabilities.

        Returns:
            Dictionary containing baseline statistics.
        """
        logger.info("Computing baseline statistics...")

        baseline = {
            "metadata": {
                "n_samples": len(data),
                "created_at": pd.Timestamp.now().isoformat(),
            },
            "features": {},
            "prediction_distribution": None
        }

        # Compute feature statistics
        for col in self.feature_columns:
            if col in data.columns:
                values = data[col].dropna()
                baseline["features"][col] = self._compute_feature_stats(values)

        # Compute prediction distribution if provided
        if prediction_probabilities is not None:
            baseline["prediction_distribution"] = self._compute_distribution_stats(
                prediction_probabilities
            )

        self.baseline = baseline
        logger.info(f"Computed baseline for {len(baseline['features'])} features")
        return baseline

    def _compute_feature_stats(self, values: pd.Series) -> dict:
        """Compute statistics for a single feature.

        Args:
            values: Series of feature values.

        Returns:
            Dictionary of statistics.
        """
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "skew": float(values.skew()),
            "kurtosis": float(values.kurtosis()),
            # Store histogram for PSI calculation
            "histogram": self._compute_histogram(values)
        }

    def _compute_histogram(
        self,
        values: pd.Series,
        n_bins: int = 10
    ) -> dict:
        """Compute histogram for PSI calculation.

        Args:
            values: Series of feature values.
            n_bins: Number of histogram bins.

        Returns:
            Dictionary with bin edges and counts.
        """
        counts, bin_edges = np.histogram(values, bins=n_bins)
        # Normalize to get proportions
        proportions = counts / counts.sum()

        return {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
            "proportions": proportions.tolist()
        }

    def _compute_distribution_stats(self, probabilities: np.ndarray) -> dict:
        """Compute statistics for prediction probability distribution.

        Args:
            probabilities: Array of prediction probabilities.

        Returns:
            Dictionary of distribution statistics.
        """
        return {
            "mean": float(np.mean(probabilities)),
            "std": float(np.std(probabilities)),
            "median": float(np.median(probabilities)),
            "histogram": self._compute_histogram(pd.Series(probabilities))
        }

    def save_baseline(self, filepath: str):
        """Save baseline statistics to a JSON file.

        Args:
            filepath: Path to save the baseline.
        """
        if self.baseline is None:
            raise ValueError("No baseline computed. Call compute_baseline first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.baseline, f, indent=2)

        logger.info(f"Saved baseline to {filepath}")

    def load_baseline(self, filepath: str) -> dict:
        """Load baseline statistics from a JSON file.

        Args:
            filepath: Path to the baseline file.

        Returns:
            The loaded baseline dictionary.
        """
        with open(filepath, "r") as f:
            self.baseline = json.load(f)

        logger.info(f"Loaded baseline from {filepath}")
        return self.baseline

    def get_baseline(self) -> dict:
        """Get the current baseline.

        Returns:
            The baseline dictionary.
        """
        if self.baseline is None:
            raise ValueError("No baseline available. Compute or load one first.")
        return self.baseline


def compute_baseline_from_csv(
    data_path: str,
    output_path: str = "models/drift_baseline.json",
    feature_columns: Optional[list[str]] = None,
    sample_size: Optional[int] = None
) -> dict:
    """Convenience function to compute baseline from a CSV file.

    Args:
        data_path: Path to the data CSV file.
        output_path: Path to save the baseline.
        feature_columns: List of feature columns to include.
        sample_size: Optional sample size for large datasets.

    Returns:
        The computed baseline dictionary.
    """
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows from dataset")

    computer = BaselineComputer(feature_columns=feature_columns)
    baseline = computer.compute_baseline(data)
    computer.save_baseline(output_path)

    return baseline
