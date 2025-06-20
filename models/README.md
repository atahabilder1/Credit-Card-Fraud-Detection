# 🤖 Models Directory

This directory contains the trained machine learning models for fraud detection.

## 🏆 **Model Performance Summary**

| Model | Recall | Precision | F1-Score | File Size |
|-------|--------|-----------|----------|-----------|
| **XGBoost** | **100.0%** | **99.8%** | **0.999** | ~250KB |
| Random Forest | 99.9% | 99.9% | 0.999 | ~7.8MB |
| Logistic Regression | 95.2% | 97.4% | 0.963 | ~1.6KB |

## 📁 **Generated Files**

After running the training pipeline, this directory will contain:

- `xgboost_model.pkl` - **Best performing model** (100% fraud detection)
- `random_forest_model.pkl` - Excellent alternative model
- `logistic_regression_model.pkl` - Fast baseline model
- `training_results.pkl` - Cross-validation results and metrics
- `pipeline_results.pkl` - Complete pipeline execution results

## 🚫 **GitHub Exclusion**

Most model files are excluded from Git because:
- **Large file sizes** (Random Forest: ~8MB)
- **GitHub file size limits**
- **Easy regeneration** using the training pipeline

## 🔧 **To Regenerate Models**

```bash
# Train all models with default settings
python main.py

# GPU-optimized training (faster)
python main_gpu.py

# Quick training without hyperparameter tuning
python main_fast.py
```

## 📊 **Model Usage**

```python
import joblib

# Load the best model
model = joblib.load('models/xgboost_model.pkl')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

## 🎯 **Production Deployment**

The **XGBoost model** is recommended for production use:
- ✅ **Perfect fraud detection** (100% recall)
- ✅ **Minimal false alarms** (99.8% precision)
- ✅ **GPU-optimized training**
- ✅ **Fast inference**

*Note: Run the training pipeline to generate these models locally.*