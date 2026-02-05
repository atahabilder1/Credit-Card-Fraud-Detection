"""
Tests for drift detection module.

Tests baseline computation, drift detection algorithms,
and alert management.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestBaselineComputer:
    """Tests for BaselineComputer class."""

    def test_baseline_computer_initialization(self, feature_columns):
        """Test baseline computer can be initialized."""
        from drift_detection.baseline import BaselineComputer
        computer = BaselineComputer(feature_columns=feature_columns)

        assert computer.feature_columns == feature_columns
        assert computer.baseline is None

    def test_compute_baseline(self, feature_columns):
        """Test baseline computation from data."""
        from drift_detection.baseline import BaselineComputer

        # Create sample data
        n_samples = 1000
        data = pd.DataFrame({
            col: np.random.randn(n_samples) for col in feature_columns
        })
        data["Amount"] = np.abs(data["Amount"]) * 100  # Make Amount positive

        computer = BaselineComputer(feature_columns=feature_columns)
        baseline = computer.compute_baseline(data)

        assert "metadata" in baseline
        assert "features" in baseline
        assert baseline["metadata"]["n_samples"] == n_samples
        assert len(baseline["features"]) == len(feature_columns)

    def test_baseline_feature_statistics(self, feature_columns):
        """Test that feature statistics are computed correctly."""
        from drift_detection.baseline import BaselineComputer

        # Create data with known statistics
        n_samples = 10000
        data = pd.DataFrame({
            col: np.random.randn(n_samples) for col in feature_columns
        })

        computer = BaselineComputer(feature_columns=feature_columns)
        baseline = computer.compute_baseline(data)

        for col in feature_columns:
            stats = baseline["features"][col]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "histogram" in stats
            # Mean should be close to 0 for standard normal
            assert -0.1 < stats["mean"] < 0.1

    def test_baseline_save_and_load(self, feature_columns):
        """Test saving and loading baseline."""
        from drift_detection.baseline import BaselineComputer

        data = pd.DataFrame({
            col: np.random.randn(100) for col in feature_columns[:3]
        })

        computer = BaselineComputer(feature_columns=feature_columns[:3])
        computer.compute_baseline(data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            computer.save_baseline(temp_path)
            assert Path(temp_path).exists()

            # Load in new computer
            computer2 = BaselineComputer()
            loaded = computer2.load_baseline(temp_path)

            assert "metadata" in loaded
            assert "features" in loaded
        finally:
            Path(temp_path).unlink()

    def test_compute_histogram(self):
        """Test histogram computation."""
        from drift_detection.baseline import BaselineComputer

        computer = BaselineComputer()
        values = pd.Series(np.random.randn(1000))
        histogram = computer._compute_histogram(values, n_bins=10)

        assert "bin_edges" in histogram
        assert "counts" in histogram
        assert "proportions" in histogram
        assert len(histogram["bin_edges"]) == 11  # n_bins + 1 edges
        assert len(histogram["counts"]) == 10
        # Proportions should sum to 1
        assert abs(sum(histogram["proportions"]) - 1.0) < 0.01


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_drift_detector_initialization(self, sample_baseline):
        """Test drift detector can be initialized."""
        from drift_detection.detector import DriftDetector
        detector = DriftDetector(baseline=sample_baseline)

        assert detector.baseline == sample_baseline
        assert detector.psi_threshold == 0.2
        assert detector.ks_threshold == 0.1

    def test_detect_no_drift(self, sample_baseline):
        """Test detecting no drift with similar distribution."""
        from drift_detection.detector import DriftDetector

        detector = DriftDetector(baseline=sample_baseline)

        # Generate data similar to baseline
        current_data = {
            "V1": np.random.randn(1000).tolist(),
            "Amount": (np.abs(np.random.randn(1000)) * 100).tolist()
        }

        result = detector.detect_drift(current_data)

        # Should not detect significant drift for similar data
        assert result.alert_level in ["none", "warning"]

    def test_detect_drift_with_shift(self, sample_baseline):
        """Test detecting drift with shifted distribution."""
        from drift_detection.detector import DriftDetector

        detector = DriftDetector(baseline=sample_baseline)

        # Generate shifted data (mean shifted by 3 std deviations)
        current_data = {
            "V1": (np.random.randn(1000) + 5.0).tolist()  # Shifted by 5
        }

        result = detector.detect_drift(current_data)

        # Should detect drift for significantly shifted data
        assert result.drift_detected is True
        assert result.feature_scores["V1"] > 0.1

    def test_calculate_psi(self, sample_baseline):
        """Test PSI calculation."""
        from drift_detection.detector import DriftDetector

        detector = DriftDetector(baseline=sample_baseline)
        baseline_histogram = sample_baseline["features"]["V1"]["histogram"]

        # Same distribution should have low PSI
        current_values = np.random.randn(1000)
        psi = detector._calculate_psi(baseline_histogram, current_values)
        assert psi < 0.5  # Should be relatively low

        # Shifted distribution should have higher PSI
        shifted_values = np.random.randn(1000) + 5.0
        psi_shifted = detector._calculate_psi(baseline_histogram, shifted_values)
        assert psi_shifted > psi

    def test_drift_result_structure(self, sample_baseline):
        """Test DriftResult has correct structure."""
        from drift_detection.detector import DriftDetector

        detector = DriftDetector(baseline=sample_baseline)
        current_data = {"V1": np.random.randn(100).tolist()}

        result = detector.detect_drift(current_data)

        assert hasattr(result, "drift_detected")
        assert hasattr(result, "overall_score")
        assert hasattr(result, "feature_scores")
        assert hasattr(result, "prediction_drift_score")
        assert hasattr(result, "alert_level")
        assert hasattr(result, "details")

    def test_check_feature_drift(self, sample_baseline):
        """Test checking drift for single feature."""
        from drift_detection.detector import DriftDetector

        detector = DriftDetector(baseline=sample_baseline)

        # Normal data
        normal_values = np.random.randn(1000)
        score, status = detector.check_feature_drift("V1", normal_values)
        assert status in ["stable", "warning"]

        # Shifted data
        shifted_values = np.random.randn(1000) + 10.0
        score, status = detector.check_feature_drift("V1", shifted_values)
        assert score > 0.1

    def test_unknown_feature(self, sample_baseline):
        """Test handling of unknown feature."""
        from drift_detection.detector import DriftDetector

        detector = DriftDetector(baseline=sample_baseline)
        score, status = detector.check_feature_drift("UnknownFeature", np.array([1, 2, 3]))

        assert score == 0.0
        assert status == "unknown"


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_alert_manager_initialization(self):
        """Test alert manager can be initialized."""
        from drift_detection.alerts import AlertManager

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            manager = AlertManager(alert_log_path=str(log_path))

            assert manager.alert_log_path == log_path

    def test_process_drift_result_no_drift(self, sample_baseline):
        """Test processing result with no drift."""
        from drift_detection.alerts import AlertManager
        from drift_detection.detector import DriftResult

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AlertManager(alert_log_path=str(Path(tmpdir) / "alerts.jsonl"))

            result = DriftResult(
                drift_detected=False,
                overall_score=0.05,
                feature_scores={"V1": 0.03},
                prediction_drift_score=0.02,
                alert_level="none",
                details={}
            )

            triggered = manager.process_drift_result(result)
            assert triggered is False

    def test_process_drift_result_with_drift(self, sample_baseline):
        """Test processing result with drift detected."""
        from drift_detection.alerts import AlertManager
        from drift_detection.detector import DriftResult

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            manager = AlertManager(alert_log_path=str(log_path))

            result = DriftResult(
                drift_detected=True,
                overall_score=0.3,
                feature_scores={"V1": 0.4, "V2": 0.2},
                prediction_drift_score=0.15,
                alert_level="critical",
                details={}
            )

            triggered = manager.process_drift_result(result)

            assert triggered is True
            assert log_path.exists()

            # Check log content
            with open(log_path) as f:
                alert = json.loads(f.readline())

            assert alert["alert_level"] == "critical"
            assert "message" in alert

    def test_get_recent_alerts(self):
        """Test retrieving recent alerts."""
        from drift_detection.alerts import AlertManager
        from drift_detection.detector import DriftResult

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "alerts.jsonl"
            manager = AlertManager(alert_log_path=str(log_path))

            # Create some alerts
            for i in range(5):
                result = DriftResult(
                    drift_detected=True,
                    overall_score=0.2 + i * 0.05,
                    feature_scores={"V1": 0.2},
                    prediction_drift_score=0.1,
                    alert_level="warning",
                    details={}
                )
                manager.process_drift_result(result)

            alerts = manager.get_recent_alerts(limit=3)
            assert len(alerts) == 3

    def test_register_handler(self):
        """Test registering custom alert handler."""
        from drift_detection.alerts import AlertManager
        from drift_detection.detector import DriftResult

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AlertManager(alert_log_path=str(Path(tmpdir) / "alerts.jsonl"))

            received_alerts = []

            def custom_handler(alert):
                received_alerts.append(alert)

            manager.register_handler(custom_handler)

            result = DriftResult(
                drift_detected=True,
                overall_score=0.3,
                feature_scores={"V1": 0.3},
                prediction_drift_score=0.15,
                alert_level="warning",
                details={}
            )
            manager.process_drift_result(result)

            assert len(received_alerts) == 1

    def test_generate_message(self):
        """Test alert message generation."""
        from drift_detection.alerts import AlertManager
        from drift_detection.detector import DriftResult

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AlertManager(alert_log_path=str(Path(tmpdir) / "alerts.jsonl"))

            result = DriftResult(
                drift_detected=True,
                overall_score=0.35,
                feature_scores={"V1": 0.4, "Amount": 0.2},
                prediction_drift_score=0.15,
                alert_level="critical",
                details={}
            )

            message = manager._generate_message(result)

            assert "CRITICAL" in message
            assert "V1" in message  # Top drifting feature
            assert "0.35" in message  # Overall score
