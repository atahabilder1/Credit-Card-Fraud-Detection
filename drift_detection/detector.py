"""
Drift detection algorithms.

Implements statistical tests to detect data and model drift:
- Population Stability Index (PSI) for distribution comparison
- Kolmogorov-Smirnov test for continuous variables
- Chi-squared test for categorical variables
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Container for drift detection results."""

    drift_detected: bool
    overall_score: float
    feature_scores: dict[str, float]
    prediction_drift_score: float
    alert_level: str  # none, warning, critical
    details: dict


class DriftDetector:
    """Detect drift in data and model predictions."""

    def __init__(
        self,
        baseline: dict,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        warning_threshold: float = 0.1,
        critical_threshold: float = 0.25
    ):
        """Initialize the drift detector.

        Args:
            baseline: Baseline statistics dictionary.
            psi_threshold: PSI threshold for significant drift.
            ks_threshold: KS test p-value threshold.
            warning_threshold: Score threshold for warning alerts.
            critical_threshold: Score threshold for critical alerts.
        """
        self.baseline = baseline
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def detect_drift(
        self,
        current_data: dict[str, list[float]],
        current_predictions: Optional[list[float]] = None
    ) -> DriftResult:
        """Detect drift in current data compared to baseline.

        Args:
            current_data: Dictionary mapping feature names to lists of values.
            current_predictions: Optional list of current prediction probabilities.

        Returns:
            DriftResult containing drift scores and alerts.
        """
        feature_scores = {}
        details = {"psi": {}, "ks": {}}

        # Check each feature
        for feature_name, current_values in current_data.items():
            if feature_name not in self.baseline.get("features", {}):
                continue

            baseline_stats = self.baseline["features"][feature_name]
            current_values = np.array(current_values)

            # Calculate PSI
            psi_score = self._calculate_psi(
                baseline_stats["histogram"],
                current_values
            )

            # Calculate KS statistic
            ks_score = self._calculate_ks_score(
                baseline_stats,
                current_values
            )

            # Combined score (average of normalized scores)
            combined_score = (psi_score + ks_score) / 2
            feature_scores[feature_name] = combined_score

            details["psi"][feature_name] = psi_score
            details["ks"][feature_name] = ks_score

        # Calculate prediction drift if available
        prediction_drift_score = 0.0
        if current_predictions and self.baseline.get("prediction_distribution"):
            prediction_drift_score = self._calculate_prediction_drift(
                self.baseline["prediction_distribution"],
                np.array(current_predictions)
            )
            details["prediction_psi"] = prediction_drift_score

        # Calculate overall score
        if feature_scores:
            overall_score = np.mean(list(feature_scores.values()))
        else:
            overall_score = 0.0

        # Determine alert level
        max_score = max(
            overall_score,
            prediction_drift_score,
            max(feature_scores.values()) if feature_scores else 0
        )

        if max_score >= self.critical_threshold:
            alert_level = "critical"
            drift_detected = True
        elif max_score >= self.warning_threshold:
            alert_level = "warning"
            drift_detected = True
        else:
            alert_level = "none"
            drift_detected = False

        return DriftResult(
            drift_detected=drift_detected,
            overall_score=overall_score,
            feature_scores=feature_scores,
            prediction_drift_score=prediction_drift_score,
            alert_level=alert_level,
            details=details
        )

    def _calculate_psi(
        self,
        baseline_histogram: dict,
        current_values: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """Calculate Population Stability Index.

        PSI measures how much a distribution has shifted.
        PSI < 0.1: No significant shift
        PSI 0.1-0.2: Moderate shift
        PSI > 0.2: Significant shift

        Args:
            baseline_histogram: Baseline histogram data.
            current_values: Current feature values.
            epsilon: Small value to avoid division by zero.

        Returns:
            PSI score.
        """
        bin_edges = np.array(baseline_histogram["bin_edges"])
        baseline_props = np.array(baseline_histogram["proportions"])

        # Calculate current histogram using same bins
        current_counts, _ = np.histogram(current_values, bins=bin_edges)
        current_props = current_counts / (current_counts.sum() + epsilon)

        # Add epsilon to avoid log(0)
        baseline_props = np.clip(baseline_props, epsilon, 1)
        current_props = np.clip(current_props, epsilon, 1)

        # PSI formula
        psi = np.sum(
            (current_props - baseline_props) *
            np.log(current_props / baseline_props)
        )

        return float(psi)

    def _calculate_ks_score(
        self,
        baseline_stats: dict,
        current_values: np.ndarray
    ) -> float:
        """Calculate Kolmogorov-Smirnov based drift score.

        Uses the KS statistic (not p-value) as a drift measure.
        Higher values indicate more drift.

        Args:
            baseline_stats: Baseline statistics for the feature.
            current_values: Current feature values.

        Returns:
            KS statistic as drift score.
        """
        # Generate synthetic baseline samples from statistics
        # This is an approximation when we don't have the original data
        baseline_mean = baseline_stats["mean"]
        baseline_std = baseline_stats["std"]

        # Use normal distribution assumption for reference
        baseline_samples = np.random.normal(
            baseline_mean,
            baseline_std,
            size=len(current_values)
        )

        # Perform KS test
        ks_statistic, _ = stats.ks_2samp(baseline_samples, current_values)

        return float(ks_statistic)

    def _calculate_prediction_drift(
        self,
        baseline_distribution: dict,
        current_predictions: np.ndarray
    ) -> float:
        """Calculate drift in prediction distribution.

        Args:
            baseline_distribution: Baseline prediction distribution stats.
            current_predictions: Current prediction probabilities.

        Returns:
            PSI score for predictions.
        """
        return self._calculate_psi(
            baseline_distribution["histogram"],
            current_predictions
        )

    def check_feature_drift(
        self,
        feature_name: str,
        current_values: np.ndarray
    ) -> tuple[float, str]:
        """Check drift for a single feature.

        Args:
            feature_name: Name of the feature.
            current_values: Current feature values.

        Returns:
            Tuple of (drift_score, status).
        """
        if feature_name not in self.baseline.get("features", {}):
            return 0.0, "unknown"

        baseline_stats = self.baseline["features"][feature_name]
        psi_score = self._calculate_psi(
            baseline_stats["histogram"],
            current_values
        )

        if psi_score >= self.psi_threshold:
            status = "drifted"
        elif psi_score >= self.psi_threshold / 2:
            status = "warning"
        else:
            status = "stable"

        return psi_score, status


def detect_drift_from_database(
    baseline_path: str,
    database,
    hours: int = 24
) -> DriftResult:
    """Convenience function to detect drift from database predictions.

    Args:
        baseline_path: Path to baseline JSON file.
        database: Database instance with get_feature_distribution method.
        hours: Number of hours of data to analyze.

    Returns:
        DriftResult with drift analysis.
    """
    import json

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    detector = DriftDetector(baseline)

    # This would need to be async in practice
    # Shown here as synchronous for simplicity
    feature_data = {}
    for feature_name in baseline.get("features", {}).keys():
        # Would call: await database.get_feature_distribution(feature_name, hours)
        pass

    # For now, return a placeholder
    return DriftResult(
        drift_detected=False,
        overall_score=0.0,
        feature_scores={},
        prediction_drift_score=0.0,
        alert_level="none",
        details={}
    )
