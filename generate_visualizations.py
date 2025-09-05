"""
Generate comprehensive visualizations and save as images.
Focus on recall, confusion matrices, and key performance metrics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, classification_report
from src.preprocess import preprocess_data
from src.utils import ensure_directory_exists

plt.style.use('default')
sns.set_palette("husl")

def create_comprehensive_visualizations():
    """Create and save all key visualizations."""

    print("GENERATING COMPREHENSIVE FRAUD DETECTION VISUALIZATIONS")
    print("=" * 60)

    # Create output directory
    viz_dir = 'visualizations'
    ensure_directory_exists(viz_dir)

    # Load test data
    print("Loading and preprocessing data...")
    df = pd.read_csv('data/creditcard_2023.csv')
    df_sample = df.groupby('Class', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 15000), random_state=42)
    ).reset_index(drop=True)

    df_sample.to_csv('data/viz_sample.csv', index=False)

    preprocessed_data = preprocess_data(
        'data/viz_sample.csv',
        test_size=0.3,
        scaler_type='standard',
        sampling_method='none',
        stratify=True
    )

    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']

    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud cases: {(y_test == 1).sum()}")
    print(f"Legitimate cases: {(y_test == 0).sum()}")

    # Load models and evaluate
    models = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }

    model_results = {}

    for model_name, model_file in models.items():
        try:
            model = joblib.load(f'models/{model_file}')
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            model_results[model_name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            print(f"✓ Loaded {model_name}")

        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")

    # 1. CONFUSION MATRICES (Most Important for Fraud Detection)
    print("\n1. Creating Confusion Matrices...")
    create_confusion_matrices(y_test, model_results, viz_dir)

    # 2. RECALL COMPARISON (Key for Fraud Detection)
    print("2. Creating Recall Analysis...")
    create_recall_analysis(y_test, model_results, viz_dir)

    # 3. ROC CURVES
    print("3. Creating ROC Curves...")
    create_roc_curves(y_test, model_results, viz_dir)

    # 4. PRECISION-RECALL CURVES
    print("4. Creating Precision-Recall Curves...")
    create_pr_curves(y_test, model_results, viz_dir)

    # 5. PERFORMANCE COMPARISON TABLE
    print("5. Creating Performance Comparison...")
    create_performance_table(y_test, model_results, viz_dir)

    # 6. DETAILED CLASSIFICATION REPORTS
    print("6. Creating Classification Reports...")
    create_classification_reports(y_test, model_results, viz_dir)

    print(f"\n✅ ALL VISUALIZATIONS SAVED TO: {viz_dir}/")
    print("Files created:")
    for file in sorted(os.listdir(viz_dir)):
        if file.endswith('.png'):
            print(f"  📊 {file}")

def create_confusion_matrices(y_test, model_results, viz_dir):
    """Create individual and combined confusion matrices."""

    # Individual confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices - Fraud Detection Performance', fontsize=16, fontweight='bold')

    for idx, (model_name, results) in enumerate(model_results.items()):
        cm = confusion_matrix(y_test, results['y_pred'])

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Legitimate', 'Fraudulent'],
                   yticklabels=['Legitimate', 'Fraudulent'])

        axes[idx].set_title(f'{model_name}\nRecall: {recall:.3f} | Precision: {precision:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                axes[idx].text(j + 0.5, i + 0.8, f'({percentage:.1f}%)',
                              ha='center', va='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(f'{viz_dir}/01_confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Individual detailed confusion matrices
    for model_name, results in model_results.items():
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, results['y_pred'])
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraudulent'],
                   yticklabels=['Legitimate', 'Fraudulent'])

        plt.title(f'{model_name} - Detailed Confusion Matrix\n' +
                 f'Recall: {recall:.3f} | Precision: {precision:.3f} | F1: {f1:.3f}\n' +
                 f'True Positives: {tp} | False Negatives: {fn} | False Positives: {fp}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Add detailed annotations
        total = cm.sum()
        labels = [['True Negatives\n(Correct Legitimate)', 'False Positives\n(False Alarms)'],
                 ['False Negatives\n(Missed Fraud)', 'True Positives\n(Caught Fraud)']]

        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.3, labels[i][j],
                        ha='center', va='center', fontsize=10, fontweight='bold')
                plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%',
                        ha='center', va='center', fontsize=12, color='red')

        filename = f"02_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(f'{viz_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

def create_recall_analysis(y_test, model_results, viz_dir):
    """Create recall-focused analysis."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Recall comparison bar chart
    recalls = []
    precisions = []
    model_names = []

    for model_name, results in model_results.items():
        cm = confusion_matrix(y_test, results['y_pred'])
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        recalls.append(recall)
        precisions.append(precision)
        model_names.append(model_name)

    # Recall bar chart
    bars1 = ax1.bar(model_names, recalls, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Recall Comparison\n(Fraud Detection Rate)', fontweight='bold')
    ax1.set_ylabel('Recall Score')
    ax1.set_ylim(0, 1.1)

    # Add value labels on bars
    for bar, recall in zip(bars1, recalls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{recall:.3f}\n({recall*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    # Precision vs Recall scatter
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (model_name, recall, precision) in enumerate(zip(model_names, recalls, precisions)):
        ax2.scatter(recall, precision, s=200, color=colors[i], alpha=0.7, label=model_name)
        ax2.annotate(f'{model_name}\n({recall:.3f}, {precision:.3f})',
                    (recall, precision), xytext=(5, 5),
                    textcoords='offset points', fontsize=10)

    ax2.set_xlabel('Recall (Fraud Detection Rate)')
    ax2.set_ylabel('Precision (Accuracy of Fraud Alerts)')
    ax2.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0.9, 1.01)
    ax2.set_ylim(0.9, 1.01)

    plt.tight_layout()
    plt.savefig(f'{viz_dir}/03_recall_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curves(y_test, model_results, viz_dir):
    """Create ROC curves."""

    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for idx, (model_name, results) in enumerate(model_results.items()):
        if results['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            from sklearn.metrics import auc
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=colors[idx], linewidth=3,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Fraud Detection Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{viz_dir}/04_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pr_curves(y_test, model_results, viz_dir):
    """Create Precision-Recall curves."""

    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for idx, (model_name, results) in enumerate(model_results.items()):
        if results['y_pred_proba'] is not None:
            precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
            from sklearn.metrics import average_precision_score
            avg_precision = average_precision_score(y_test, results['y_pred_proba'])

            plt.plot(recall, precision, color=colors[idx], linewidth=3,
                    label=f'{model_name} (AP = {avg_precision:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Fraud Detection Rate)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Fraud Detection Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{viz_dir}/05_precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_table(y_test, model_results, viz_dir):
    """Create performance comparison table."""

    # Calculate all metrics
    performance_data = []

    for model_name, results in model_results.items():
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        cm = confusion_matrix(y_test, results['y_pred'])
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_test, results['y_pred'])
        precision = precision_score(y_test, results['y_pred'], zero_division=0)
        recall = recall_score(y_test, results['y_pred'], zero_division=0)
        f1 = f1_score(y_test, results['y_pred'], zero_division=0)

        if results['y_pred_proba'] is not None:
            roc_auc = roc_auc_score(y_test, results['y_pred_proba'])
        else:
            roc_auc = 0.0

        performance_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'True Positives': tp,
            'False Negatives': fn,
            'False Positives': fp,
            'Fraud Detection Rate': f'{recall*100:.1f}%'
        })

    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    df = pd.DataFrame(performance_data)

    # Create table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Accuracy']:.3f}",
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f"{row['F1-Score']:.3f}",
            f"{row['ROC-AUC']:.3f}",
            f"{row['True Positives']}",
            f"{row['False Negatives']}",
            f"{row['False Positives']}"
        ])

    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'TP', 'FN', 'FP']

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color code the best values
    for i in range(len(table_data)):
        # Highlight best recall (most important for fraud detection)
        if df.iloc[i]['Recall'] == df['Recall'].max():
            table[(i+1, 3)].set_facecolor('#90EE90')  # Light green for best recall

    plt.title('Performance Comparison Table - Fraud Detection Models',
              fontsize=14, fontweight='bold', pad=20)

    plt.savefig(f'{viz_dir}/06_performance_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_classification_reports(y_test, model_results, viz_dir):
    """Create detailed classification reports."""

    for model_name, results in model_results.items():
        # Generate classification report
        report = classification_report(y_test, results['y_pred'],
                                     target_names=['Legitimate', 'Fraudulent'],
                                     output_dict=True)

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data for heatmap
        classes = ['Legitimate', 'Fraudulent']
        metrics = ['precision', 'recall', 'f1-score']

        data = []
        for cls in classes:
            row = []
            for metric in metrics:
                row.append(report[cls][metric])
            data.append(row)

        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=metrics, yticklabels=classes,
                   ax=ax, cbar_kws={'label': 'Score'})

        ax.set_title(f'{model_name} - Detailed Classification Report',
                    fontweight='bold', fontsize=14)

        # Add support information
        support_text = f"Support - Legitimate: {report['Legitimate']['support']}, " + \
                      f"Fraudulent: {report['Fraudulent']['support']}"
        ax.text(0.5, -0.1, support_text, transform=ax.transAxes,
               ha='center', fontsize=10)

        filename = f"07_classification_report_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(f'{viz_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    create_comprehensive_visualizations()