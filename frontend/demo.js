// Credit Card Fraud Detection - Demo JavaScript

const API_BASE = window.location.origin;

// Example transactions
const LEGITIMATE_EXAMPLE = {
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
};

const FRAUD_EXAMPLE = {
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
};

// Generate random transaction
function generateRandomTransaction(isFraud = false) {
    const stats = isFraud ? {
        vMean: -2.0,
        vStd: 3.0,
        amountMean: 150,
        amountStd: 500
    } : {
        vMean: 0.0,
        vStd: 1.5,
        amountMean: 88,
        amountStd: 250
    };

    const transaction = {};
    for (let i = 1; i <= 28; i++) {
        transaction[`V${i}`] = randomGaussian(stats.vMean, stats.vStd);
    }
    transaction.Amount = Math.max(0, randomGaussian(stats.amountMean, stats.amountStd));

    return transaction;
}

function randomGaussian(mean, std) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z * std;
}

// API calls
async function makePrediction(transaction) {
    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(transaction)
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    return response.json();
}

async function fetchStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (response.ok) {
            return response.json();
        }
    } catch (e) {
        console.log('Stats fetch failed:', e);
    }
    return null;
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        return response.ok;
    } catch (e) {
        return false;
    }
}

// Display functions
function showLoading() {
    const resultDiv = document.getElementById('demo-result');
    resultDiv.innerHTML = `
        <div class="result-placeholder">
            <div class="loading"></div>
            <p style="margin-top: 16px;">Processing transaction...</p>
        </div>
    `;
}

function showResult(result, transactionType) {
    const resultDiv = document.getElementById('demo-result');
    const isFraud = result.prediction === 1;
    const predictionClass = isFraud ? 'fraud' : 'legitimate';
    const probabilityPercent = (result.fraud_probability * 100).toFixed(2);

    resultDiv.innerHTML = `
        <div class="result-content">
            <div class="result-header">
                <div class="result-prediction ${predictionClass}">
                    ${isFraud ? '🚨 FRAUD DETECTED' : '✅ LEGITIMATE'}
                </div>
                <div class="result-probability">
                    ${probabilityPercent}% fraud probability
                </div>
            </div>
            <div class="result-details">
                <div class="result-detail">
                    <span class="detail-label">Transaction ID</span>
                    <span class="detail-value">${result.transaction_id}</span>
                </div>
                <div class="result-detail">
                    <span class="detail-label">Input Type</span>
                    <span class="detail-value">${transactionType}</span>
                </div>
                <div class="result-detail">
                    <span class="detail-label">Confidence</span>
                    <span class="detail-value">${(result.confidence * 100).toFixed(2)}%</span>
                </div>
                <div class="result-detail">
                    <span class="detail-label">Processing Time</span>
                    <span class="detail-value">${result.processing_time_ms.toFixed(2)} ms</span>
                </div>
                <div class="result-detail">
                    <span class="detail-label">Model Version</span>
                    <span class="detail-value">${result.model_version}</span>
                </div>
                <div class="result-detail">
                    <span class="detail-label">Timestamp</span>
                    <span class="detail-value">${new Date(result.timestamp).toLocaleString()}</span>
                </div>
            </div>
        </div>
    `;
}

function showError(message) {
    const resultDiv = document.getElementById('demo-result');
    resultDiv.innerHTML = `
        <div class="result-placeholder" style="color: #ef4444;">
            <p>❌ Error: ${message}</p>
            <p style="margin-top: 8px; font-size: 0.875rem;">Make sure the API is running.</p>
        </div>
    `;
}

// Button handlers
async function testLegitimate() {
    showLoading();
    try {
        const result = await makePrediction(LEGITIMATE_EXAMPLE);
        showResult(result, 'Legitimate Example');
    } catch (e) {
        showError(e.message);
    }
}

async function testFraud() {
    showLoading();
    try {
        const result = await makePrediction(FRAUD_EXAMPLE);
        showResult(result, 'Fraud Example');
    } catch (e) {
        showError(e.message);
    }
}

async function testRandom() {
    showLoading();
    try {
        const isFraud = Math.random() < 0.1; // 10% chance of fraud
        const transaction = generateRandomTransaction(isFraud);
        const result = await makePrediction(transaction);
        showResult(result, `Random (${isFraud ? 'fraud-like' : 'legitimate-like'})`);
    } catch (e) {
        showError(e.message);
    }
}

// Update stats periodically
async function updateStats() {
    const stats = await fetchStats();
    const healthy = await checkHealth();

    // Update API status
    const statusEl = document.getElementById('api-status');
    if (healthy) {
        statusEl.textContent = '● UP';
        statusEl.style.color = '#10b981';
    } else {
        statusEl.textContent = '● DOWN';
        statusEl.style.color = '#ef4444';
    }

    // Update prediction stats
    if (stats) {
        document.getElementById('total-predictions').textContent =
            stats.total_predictions?.toLocaleString() || '0';
        document.getElementById('fraud-detected').textContent =
            stats.fraud_detected?.toLocaleString() || '0';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initial stats update
    updateStats();

    // Update stats every 30 seconds
    setInterval(updateStats, 30000);

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'l' || e.key === 'L') {
            testLegitimate();
        } else if (e.key === 'f' || e.key === 'F') {
            testFraud();
        } else if (e.key === 'r' || e.key === 'R') {
            testRandom();
        }
    });
});
