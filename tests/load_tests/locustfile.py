"""
Locust load testing for the Fraud Detection API.

Run with: locust -f tests/load_tests/locustfile.py --host=http://localhost:8000

This simulates realistic user load to test API performance under stress.
"""

import random

from locust import HttpUser, between, task


# Sample transactions for testing
LEGITIMATE_TRANSACTION = {
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62
}

FRAUD_TRANSACTION = {
    "V1": -2.3122265423263,
    "V2": 1.95199201064158,
    "V3": -1.60985073229769,
    "V4": 3.9979055875468,
    "V5": -0.522187864667764,
    "V6": -1.42654531920595,
    "V7": -2.53738730624579,
    "V8": 1.39165724829804,
    "V9": -2.77008927719433,
    "V10": -2.77227214465915,
    "V11": 3.20203320709635,
    "V12": -2.89990738849473,
    "V13": -0.595221881324605,
    "V14": -4.28925378244217,
    "V15": 0.389724120274487,
    "V16": -1.14074717980657,
    "V17": -2.83005567450437,
    "V18": -0.0168224681808257,
    "V19": 0.416955705037907,
    "V20": 0.126910559061474,
    "V21": 0.517232370861764,
    "V22": -0.0350493686052974,
    "V23": -0.465211076182388,
    "V24": 0.320198198514526,
    "V25": 0.0445191674731724,
    "V26": 0.177839798284401,
    "V27": 0.261145002567677,
    "V28": -0.143275874698919,
    "Amount": 0.0
}


def generate_random_transaction():
    """Generate a random transaction with slight variations."""
    base = LEGITIMATE_TRANSACTION.copy()
    # Add small random noise to make each request unique
    for key in base:
        if key.startswith("V"):
            base[key] += random.gauss(0, 0.1)
        elif key == "Amount":
            base[key] = max(0, base[key] + random.gauss(0, 50))
    return base


class FraudDetectionUser(HttpUser):
    """Simulates a user interacting with the Fraud Detection API."""

    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)

    @task(10)
    def predict_single(self):
        """Test single prediction endpoint (most common operation)."""
        # 95% legitimate, 5% fraud
        if random.random() < 0.05:
            transaction = FRAUD_TRANSACTION.copy()
        else:
            transaction = generate_random_transaction()

        self.client.post(
            "/predict",
            json=transaction,
            name="/predict"
        )

    @task(2)
    def predict_batch_small(self):
        """Test small batch prediction (10 transactions)."""
        transactions = [generate_random_transaction() for _ in range(10)]
        self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
            name="/predict/batch (10)"
        )

    @task(1)
    def predict_batch_large(self):
        """Test larger batch prediction (100 transactions)."""
        transactions = [generate_random_transaction() for _ in range(100)]
        self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
            name="/predict/batch (100)"
        )

    @task(5)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health", name="/health")

    @task(2)
    def readiness_check(self):
        """Test readiness check endpoint."""
        self.client.get("/health/ready", name="/health/ready")

    @task(1)
    def model_info(self):
        """Test model info endpoint."""
        self.client.get("/model/info", name="/model/info")

    @task(1)
    def get_stats(self):
        """Test statistics endpoint."""
        self.client.get("/stats", name="/stats")

    @task(1)
    def get_drift_status(self):
        """Test drift status endpoint."""
        self.client.get("/drift/status", name="/drift/status")


class HighLoadUser(HttpUser):
    """Simulates high-frequency prediction requests."""

    wait_time = between(0.1, 0.5)  # Very short wait time

    @task
    def rapid_predict(self):
        """Rapid single predictions."""
        self.client.post(
            "/predict",
            json=generate_random_transaction(),
            name="/predict (high load)"
        )


class BatchProcessingUser(HttpUser):
    """Simulates batch processing workload."""

    wait_time = between(5, 15)

    @task(3)
    def batch_medium(self):
        """Medium batch (50 transactions)."""
        transactions = [generate_random_transaction() for _ in range(50)]
        self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
            name="/predict/batch (50)"
        )

    @task(1)
    def batch_large(self):
        """Large batch (500 transactions)."""
        transactions = [generate_random_transaction() for _ in range(500)]
        self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
            name="/predict/batch (500)"
        )

    @task(1)
    def batch_max(self):
        """Maximum batch (1000 transactions)."""
        transactions = [generate_random_transaction() for _ in range(1000)]
        self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
            name="/predict/batch (1000)"
        )
