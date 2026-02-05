"""
Alert management for drift detection.

Handles alert triggering, logging, and notifications
when drift is detected.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from drift_detection.detector import DriftResult

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage drift alerts and notifications."""

    def __init__(
        self,
        alert_log_path: str = "logs/drift_alerts.jsonl",
        webhook_url: Optional[str] = None,
        email_config: Optional[dict] = None
    ):
        """Initialize the alert manager.

        Args:
            alert_log_path: Path to the alert log file.
            webhook_url: Optional webhook URL for notifications.
            email_config: Optional email configuration dictionary.
        """
        self.alert_log_path = Path(alert_log_path)
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.webhook_url = webhook_url
        self.email_config = email_config
        self._handlers: list[Callable[[dict], None]] = []

    def process_drift_result(self, result: DriftResult) -> bool:
        """Process a drift detection result and trigger alerts if needed.

        Args:
            result: DriftResult from drift detection.

        Returns:
            True if an alert was triggered, False otherwise.
        """
        if not result.drift_detected:
            return False

        alert = self._create_alert(result)
        self._log_alert(alert)
        self._notify(alert)

        return True

    def _create_alert(self, result: DriftResult) -> dict:
        """Create an alert dictionary from drift result.

        Args:
            result: DriftResult object.

        Returns:
            Alert dictionary.
        """
        # Find top drifting features
        top_features = sorted(
            result.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_level": result.alert_level,
            "drift_detected": result.drift_detected,
            "overall_score": result.overall_score,
            "prediction_drift_score": result.prediction_drift_score,
            "top_drifting_features": [
                {"feature": f, "score": s} for f, s in top_features
            ],
            "total_features_checked": len(result.feature_scores),
            "features_with_drift": sum(
                1 for s in result.feature_scores.values() if s > 0.1
            ),
            "message": self._generate_message(result)
        }

        return alert

    def _generate_message(self, result: DriftResult) -> str:
        """Generate a human-readable alert message.

        Args:
            result: DriftResult object.

        Returns:
            Alert message string.
        """
        if result.alert_level == "critical":
            prefix = "CRITICAL DRIFT DETECTED"
        elif result.alert_level == "warning":
            prefix = "Warning: Drift detected"
        else:
            prefix = "Drift check complete"

        top_feature = max(
            result.feature_scores.items(),
            key=lambda x: x[1],
            default=("none", 0)
        )

        message = (
            f"{prefix}. "
            f"Overall drift score: {result.overall_score:.3f}. "
            f"Top drifting feature: {top_feature[0]} ({top_feature[1]:.3f}). "
            f"Prediction drift: {result.prediction_drift_score:.3f}."
        )

        return message

    def _log_alert(self, alert: dict):
        """Log alert to file.

        Args:
            alert: Alert dictionary.
        """
        with open(self.alert_log_path, "a") as f:
            f.write(json.dumps(alert) + "\n")

        logger.warning(f"Drift alert: {alert['message']}")

    def _notify(self, alert: dict):
        """Send notifications for the alert.

        Args:
            alert: Alert dictionary.
        """
        # Call registered handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        # Send webhook if configured
        if self.webhook_url:
            self._send_webhook(alert)

        # Send email if configured and critical
        if self.email_config and alert["alert_level"] == "critical":
            self._send_email(alert)

    def _send_webhook(self, alert: dict):
        """Send alert to webhook URL.

        Args:
            alert: Alert dictionary.
        """
        try:
            import requests
            response = requests.post(
                self.webhook_url,
                json=alert,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Webhook notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    def _send_email(self, alert: dict):
        """Send email notification.

        Args:
            alert: Alert dictionary.
        """
        if not self.email_config:
            return

        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(
                f"Drift Alert\n\n"
                f"Level: {alert['alert_level']}\n"
                f"Time: {alert['timestamp']}\n"
                f"Message: {alert['message']}\n\n"
                f"Details:\n{json.dumps(alert, indent=2)}"
            )
            msg["Subject"] = f"[{alert['alert_level'].upper()}] Fraud Detection Model Drift"
            msg["From"] = self.email_config.get("from_email")
            msg["To"] = self.email_config.get("to_email")

            with smtplib.SMTP(
                self.email_config.get("smtp_host", "localhost"),
                self.email_config.get("smtp_port", 25)
            ) as server:
                if self.email_config.get("smtp_user"):
                    server.login(
                        self.email_config["smtp_user"],
                        self.email_config["smtp_password"]
                    )
                server.send_message(msg)

            logger.info("Email notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def register_handler(self, handler: Callable[[dict], None]):
        """Register a custom alert handler.

        Args:
            handler: Callable that takes an alert dictionary.
        """
        self._handlers.append(handler)

    def get_recent_alerts(self, limit: int = 100) -> list[dict]:
        """Get recent alerts from the log file.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of alert dictionaries.
        """
        if not self.alert_log_path.exists():
            return []

        alerts = []
        with open(self.alert_log_path, "r") as f:
            for line in f:
                if line.strip():
                    alerts.append(json.loads(line))

        return alerts[-limit:]

    def clear_alerts(self):
        """Clear the alert log file."""
        if self.alert_log_path.exists():
            self.alert_log_path.unlink()
        logger.info("Alert log cleared")


# Prometheus metrics integration
def update_prometheus_metrics(result: DriftResult):
    """Update Prometheus metrics with drift detection results.

    Args:
        result: DriftResult from drift detection.
    """
    try:
        from api.monitoring import set_drift_alert, update_drift_scores

        update_drift_scores(
            overall=result.overall_score,
            feature_scores=result.feature_scores,
            prediction_drift=result.prediction_drift_score
        )

        set_drift_alert("warning", result.alert_level == "warning")
        set_drift_alert("critical", result.alert_level == "critical")

    except ImportError:
        logger.debug("Prometheus metrics not available")
