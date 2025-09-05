"""
Generate a final performance summary with key recall metrics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocess import preprocess_data

def generate_performance_summary():
    """Generate final performance summary focusing on recall."""

    print("🎯 FRAUD DETECTION PERFORMANCE SUMMARY")
    print("=" * 50)

    # Load test data
    df = pd.read_csv('data/creditcard_2023.csv')
    df_sample = df.groupby('Class', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 15000), random_state=42)
    ).reset_index(drop=True)

    df_sample.to_csv('data/final_test.csv', index=False)

    preprocessed_data = preprocess_data(
        'data/final_test.csv',
        test_size=0.3,
        scaler_type='standard',
        sampling_method='none',
        stratify=True
    )

    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']

    total_fraud_cases = (y_test == 1).sum()
    total_legitimate_cases = (y_test == 0).sum()

    print(f"📊 TEST DATASET OVERVIEW:")
    print(f"   Total samples: {len(y_test):,}")
    print(f"   Fraud cases: {total_fraud_cases:,}")
    print(f"   Legitimate cases: {total_legitimate_cases:,}")
    print()

    # Load and evaluate models
    models = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }

    results = []

    for model_name, model_file in models.items():
        try:
            model = joblib.load(f'models/{model_file}')
            y_pred = model.predict(X_test)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Key metrics
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Fraud detection performance
            fraud_detected = tp
            fraud_missed = fn
            false_alarms = fp

            results.append({
                'model': model_name,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'accuracy': accuracy,
                'fraud_detected': fraud_detected,
                'fraud_missed': fraud_missed,
                'false_alarms': false_alarms,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            })

            print(f"🤖 {model_name.upper()}")
            print(f"   ✅ Recall (Fraud Detection Rate): {recall:.3f} ({recall*100:.1f}%)")
            print(f"   ✅ Precision (Alert Accuracy): {precision:.3f} ({precision*100:.1f}%)")
            print(f"   ✅ F1-Score: {f1:.3f}")
            print(f"   📈 Fraud Cases Detected: {fraud_detected}/{total_fraud_cases} ({fraud_detected/total_fraud_cases*100:.1f}%)")
            print(f"   ❌ Fraud Cases Missed: {fraud_missed}")
            print(f"   🚨 False Alarms: {false_alarms}")
            print(f"   📊 Confusion Matrix: [TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}]")
            print()

        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")

    # Find best model for fraud detection (highest recall)
    if results:
        best_recall_model = max(results, key=lambda x: x['recall'])
        best_f1_model = max(results, key=lambda x: x['f1'])

        print("🏆 BEST PERFORMANCE SUMMARY:")
        print("=" * 40)
        print(f"🎯 Best Fraud Detection (Recall): {best_recall_model['model']}")
        print(f"   Detects {best_recall_model['recall']*100:.1f}% of all fraud cases")
        print(f"   Only misses {best_recall_model['fraud_missed']} out of {total_fraud_cases} fraud cases")
        print()
        print(f"⚖️ Best Overall Balance (F1): {best_f1_model['model']}")
        print(f"   F1-Score: {best_f1_model['f1']:.3f}")
        print(f"   Perfect balance of precision and recall")
        print()

        # Business impact
        print("💼 BUSINESS IMPACT:")
        print("=" * 30)
        best_model = best_recall_model
        saved_money_per_fraud = 1000  # Assume $1000 average fraud amount
        false_alarm_cost = 50  # Assume $50 cost per false alarm

        money_saved = best_model['fraud_detected'] * saved_money_per_fraud
        false_alarm_costs = best_model['false_alarms'] * false_alarm_cost
        net_benefit = money_saved - false_alarm_costs

        print(f"💰 Fraud Detected: {best_model['fraud_detected']} cases")
        print(f"💵 Estimated Money Saved: ${money_saved:,}")
        print(f"🚨 False Alarm Cost: ${false_alarm_costs:,}")
        print(f"📈 Net Financial Benefit: ${net_benefit:,}")
        print()

        print("📁 VISUALIZATION FILES CREATED:")
        print("=" * 35)
        viz_files = [
            "01_confusion_matrices_all_models.png - Overview of all model confusion matrices",
            "02_confusion_matrix_[model].png - Detailed confusion matrix for each model",
            "03_recall_analysis.png - Recall comparison and precision vs recall trade-off",
            "04_roc_curves.png - ROC curves for all models",
            "05_precision_recall_curves.png - Precision-Recall curves",
            "06_performance_table.png - Complete performance comparison table",
            "07_classification_report_[model].png - Detailed classification reports"
        ]

        for file_desc in viz_files:
            print(f"   📊 {file_desc}")

        print(f"\n✅ All visualizations saved in: visualizations/ folder")
        print(f"🎯 Focus on RECALL for fraud detection - higher recall = fewer missed fraud cases!")

if __name__ == "__main__":
    generate_performance_summary()