#!/usr/bin/env python3
"""
Traffic simulator for generating realistic transaction data.

Runs continuously to keep Grafana dashboards active with real-time data.
Generates transactions at configurable intervals with realistic patterns.

Usage:
    python -m scripts.traffic_simulator

Environment Variables:
    API_URL: API endpoint URL (default: http://localhost:8000)
    SIMULATION_INTERVAL_MIN: Minimum seconds between requests (default: 5)
    SIMULATION_INTERVAL_MAX: Maximum seconds between requests (default: 30)
    FRAUD_RATE: Percentage of fraudulent transactions (default: 0.02)
"""

import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrafficSimulator:
    """Generate simulated transaction traffic."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        interval_min: int = 5,
        interval_max: int = 30,
        fraud_rate: float = 0.02
    ):
        """Initialize the traffic simulator.

        Args:
            api_url: Base URL of the fraud detection API.
            interval_min: Minimum seconds between transactions.
            interval_max: Maximum seconds between transactions.
            fraud_rate: Probability of generating a fraudulent transaction.
        """
        self.api_url = api_url.rstrip("/")
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.fraud_rate = fraud_rate
        self._running = True

        # Statistics based on typical credit card transaction data
        # These are approximate values for PCA-transformed features
        self.feature_stats = {
            "legitimate": {
                "V1": {"mean": 0.0, "std": 1.5},
                "V2": {"mean": 0.0, "std": 1.5},
                "V3": {"mean": 0.0, "std": 1.5},
                "V4": {"mean": 0.0, "std": 1.5},
                "V5": {"mean": 0.0, "std": 1.5},
                "V6": {"mean": 0.0, "std": 1.5},
                "V7": {"mean": 0.0, "std": 1.5},
                "V8": {"mean": 0.0, "std": 1.5},
                "V9": {"mean": 0.0, "std": 1.5},
                "V10": {"mean": 0.0, "std": 1.5},
                "V11": {"mean": 0.0, "std": 1.5},
                "V12": {"mean": 0.0, "std": 1.5},
                "V13": {"mean": 0.0, "std": 1.5},
                "V14": {"mean": 0.0, "std": 1.5},
                "V15": {"mean": 0.0, "std": 1.5},
                "V16": {"mean": 0.0, "std": 1.5},
                "V17": {"mean": 0.0, "std": 1.5},
                "V18": {"mean": 0.0, "std": 1.5},
                "V19": {"mean": 0.0, "std": 1.5},
                "V20": {"mean": 0.0, "std": 1.5},
                "V21": {"mean": 0.0, "std": 0.8},
                "V22": {"mean": 0.0, "std": 0.8},
                "V23": {"mean": 0.0, "std": 0.6},
                "V24": {"mean": 0.0, "std": 0.5},
                "V25": {"mean": 0.0, "std": 0.5},
                "V26": {"mean": 0.0, "std": 0.5},
                "V27": {"mean": 0.0, "std": 0.4},
                "V28": {"mean": 0.0, "std": 0.3},
                "Amount": {"mean": 88.0, "std": 250.0, "min": 0, "max": 5000}
            },
            "fraud": {
                "V1": {"mean": -4.0, "std": 3.0},
                "V2": {"mean": 3.0, "std": 3.0},
                "V3": {"mean": -5.0, "std": 3.0},
                "V4": {"mean": 4.0, "std": 2.0},
                "V5": {"mean": -2.0, "std": 3.0},
                "V6": {"mean": -1.5, "std": 2.0},
                "V7": {"mean": -5.0, "std": 3.0},
                "V8": {"mean": 0.5, "std": 4.0},
                "V9": {"mean": -3.0, "std": 2.5},
                "V10": {"mean": -5.0, "std": 3.0},
                "V11": {"mean": 2.0, "std": 2.5},
                "V12": {"mean": -6.0, "std": 4.0},
                "V13": {"mean": 0.0, "std": 2.0},
                "V14": {"mean": -8.0, "std": 4.0},
                "V15": {"mean": 0.0, "std": 2.0},
                "V16": {"mean": -4.0, "std": 3.0},
                "V17": {"mean": -6.0, "std": 4.0},
                "V18": {"mean": -1.5, "std": 2.5},
                "V19": {"mean": 0.5, "std": 2.0},
                "V20": {"mean": 0.2, "std": 1.0},
                "V21": {"mean": 0.5, "std": 2.0},
                "V22": {"mean": 0.2, "std": 1.5},
                "V23": {"mean": -0.2, "std": 1.0},
                "V24": {"mean": 0.0, "std": 0.6},
                "V25": {"mean": 0.0, "std": 0.6},
                "V26": {"mean": 0.0, "std": 0.6},
                "V27": {"mean": 0.3, "std": 0.8},
                "V28": {"mean": 0.1, "std": 0.5},
                "Amount": {"mean": 150.0, "std": 500.0, "min": 0, "max": 2500}
            }
        }

        # Track statistics
        self.stats = {
            "total_sent": 0,
            "fraud_sent": 0,
            "legitimate_sent": 0,
            "errors": 0,
            "start_time": None
        }

    def generate_transaction(self, is_fraud: bool = False) -> dict:
        """Generate a realistic transaction.

        Args:
            is_fraud: Whether to generate a fraudulent transaction.

        Returns:
            Dictionary of transaction features.
        """
        stats = self.feature_stats["fraud" if is_fraud else "legitimate"]
        transaction = {}

        for feature, params in stats.items():
            if feature == "Amount":
                # Amount follows log-normal distribution
                value = abs(random.gauss(params["mean"], params["std"]))
                value = max(params.get("min", 0), min(value, params.get("max", 10000)))
            else:
                value = random.gauss(params["mean"], params["std"])

            transaction[feature] = round(value, 10)

        return transaction

    def send_transaction(self, transaction: dict) -> Optional[dict]:
        """Send a transaction to the API.

        Args:
            transaction: Transaction data dictionary.

        Returns:
            API response or None if failed.
        """
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=transaction,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error - API might not be ready")
            return None
        except requests.exceptions.Timeout:
            logger.warning("Request timeout")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def wait_for_api(self, max_retries: int = 30, retry_interval: int = 10):
        """Wait for the API to become available.

        Args:
            max_retries: Maximum number of retry attempts.
            retry_interval: Seconds between retries.
        """
        logger.info(f"Waiting for API at {self.api_url}...")

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.api_url}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("API is ready!")
                    return True
            except Exception:
                pass

            logger.info(f"API not ready, retrying in {retry_interval}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_interval)

        logger.error("API did not become available")
        return False

    def run(self):
        """Run the traffic simulation loop."""
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("=" * 60)
        logger.info("TRAFFIC SIMULATOR STARTING")
        logger.info("=" * 60)
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Interval: {self.interval_min}-{self.interval_max} seconds")
        logger.info(f"Fraud rate: {self.fraud_rate * 100:.1f}%")
        logger.info("=" * 60)

        # Wait for API to be ready
        if not self.wait_for_api():
            logger.error("Exiting - API not available")
            return

        self.stats["start_time"] = datetime.now()
        logger.info("Starting traffic generation...")

        while self._running:
            try:
                # Determine if this should be a fraud transaction
                is_fraud = random.random() < self.fraud_rate

                # Generate and send transaction
                transaction = self.generate_transaction(is_fraud=is_fraud)
                result = self.send_transaction(transaction)

                # Update statistics
                self.stats["total_sent"] += 1
                if result:
                    if is_fraud:
                        self.stats["fraud_sent"] += 1
                    else:
                        self.stats["legitimate_sent"] += 1

                    predicted = result.get("prediction_label", "unknown")
                    prob = result.get("fraud_probability", 0)

                    logger.info(
                        f"Transaction #{self.stats['total_sent']}: "
                        f"Generated={'FRAUD' if is_fraud else 'legit'}, "
                        f"Predicted={predicted}, "
                        f"Probability={prob:.4f}"
                    )
                else:
                    self.stats["errors"] += 1

                # Log periodic summary
                if self.stats["total_sent"] % 100 == 0:
                    self._log_summary()

                # Wait before next transaction
                interval = random.randint(self.interval_min, self.interval_max)
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                self.stats["errors"] += 1
                time.sleep(5)

        self._log_summary()
        logger.info("Traffic simulator stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self._running = False

    def _log_summary(self):
        """Log a summary of simulation statistics."""
        if self.stats["start_time"]:
            runtime = datetime.now() - self.stats["start_time"]
            rate = self.stats["total_sent"] / max(runtime.total_seconds(), 1) * 60
        else:
            rate = 0

        logger.info("-" * 40)
        logger.info("SIMULATION SUMMARY")
        logger.info(f"  Total transactions: {self.stats['total_sent']}")
        logger.info(f"  Fraud sent: {self.stats['fraud_sent']}")
        logger.info(f"  Legitimate sent: {self.stats['legitimate_sent']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Rate: {rate:.2f} transactions/minute")
        logger.info("-" * 40)


def main():
    """Main entry point."""
    # Get configuration from environment
    api_url = os.getenv("API_URL", "http://localhost:8000")
    interval_min = int(os.getenv("SIMULATION_INTERVAL_MIN", "5"))
    interval_max = int(os.getenv("SIMULATION_INTERVAL_MAX", "30"))
    fraud_rate = float(os.getenv("FRAUD_RATE", "0.02"))

    simulator = TrafficSimulator(
        api_url=api_url,
        interval_min=interval_min,
        interval_max=interval_max,
        fraud_rate=fraud_rate
    )

    simulator.run()


if __name__ == "__main__":
    main()
