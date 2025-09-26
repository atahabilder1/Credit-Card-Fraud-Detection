"""
Quick evaluation of trained models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.preprocess import preprocess_data

def evaluate_saved_models():
    """Evaluate the saved models on test data."""

    print("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION")
    print("=" * 55)

    # Load test data (use sample for quick evaluation)
    print("Loading and preprocessing data...")

    # Create a smaller sample for quick evaluation
    df = pd.read_csv('data/creditcard_2023.csv')
    df_sample = df.groupby('Class', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 10000), random_state=42)
    ).reset_index(drop=True)

    df_sample.to_csv('data/eval_sample.csv', index=False)

    # Preprocess
    preprocessed_data = preprocess_data(
        'data/eval_sample.csv',
        test_size=0.3,
        scaler_type='standard',
        sampling_method='none',  # No resampling for evaluation
        stratify=True
    )

    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']

    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud cases: {(y_test == 1).sum()}")
    print(f"Legitimate cases: {(y_test == 0).sum()}")
    print()

    # Load and evaluate each model
    models = ['logistic_regression', 'random_forest', 'xgboost']
    results = {}

    for model_name in models:
        try:
            model_path = f'models/{model_name}_model.pkl'
            model = joblib.load(model_path)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                roc_auc = None

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }

            print(f"{model_name.upper().replace('_', ' ')} RESULTS:")
            print("-" * 35)
            print(f"Accuracy:  {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"F1-Score:  {f1:.3f}")
            if roc_auc:
                print(f"ROC-AUC:   {roc_auc:.3f}")
            print(f"True Positives:  {tp} (fraud detected)")
            print(f"False Positives: {fp} (false alarms)")
            print(f"False Negatives: {fn} (missed fraud)")
            print(f"True Negatives:  {tn} (correct legitimate)")
            print()

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    # Model comparison
    if results:
        print("MODEL COMPARISON SUMMARY:")
        print("=" * 50)
        print(f"{'Model':<20} {'F1':<6} {'Precision':<9} {'Recall':<6} {'ROC-AUC':<7}")
        print("-" * 50)

        for model_name, metrics in results.items():
            model_display = model_name.replace('_', ' ').title()
            f1 = metrics['f1_score']
            precision = metrics['precision']
            recall = metrics['recall']
            roc_auc = metrics['roc_auc'] if metrics['roc_auc'] else 0.0

            print(f"{model_display:<20} {f1:<6.3f} {precision:<9.3f} {recall:<6.3f} {roc_auc:<7.3f}")

        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_f1 = results[best_model]['f1_score']

        print()
        print(f"🏆 BEST MODEL: {best_model.replace('_', ' ').title()}")
        print(f"   F1-Score: {best_f1:.3f}")
        print(f"   Fraud Detection Rate: {results[best_model]['recall']*100:.1f}%")
        print(f"   Precision: {results[best_model]['precision']*100:.1f}%")

        # Practical interpretation
        total_fraud = (y_test == 1).sum()
        detected_fraud = results[best_model]['tp']
        false_alarms = results[best_model]['fp']

        print(f"\n📊 PRACTICAL PERFORMANCE:")
        print(f"   Out of {total_fraud} fraud cases, model detected {detected_fraud}")
        print(f"   Generated {false_alarms} false alarms")
        print(f"   Missed {total_fraud - detected_fraud} fraud cases")

if __name__ == "__main__":
    evaluate_saved_models()