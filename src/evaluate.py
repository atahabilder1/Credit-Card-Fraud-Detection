"""
Model evaluation module for credit card fraud detection.
Provides comprehensive evaluation metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import learning_curve
import warnings
from typing import Dict, Any, List, Tuple, Optional

warnings.filterwarnings('ignore')

class FraudDetectionEvaluator:
    """
    Class for evaluating fraud detection models.
    """

    def __init__(self):
        """
        Initialize the evaluator.
        """
        self.evaluation_results = {}

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a fraud detection model.

        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            model_name (str): Name of the model for reporting

        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        if model_name is None:
            model_name = type(model).__name__

        print(f"Evaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)

        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate AUC metrics if probabilities are available
        roc_auc = None
        pr_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'mcc': mcc,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'confusion_matrix': cm,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }

        self.evaluation_results[model_name] = results

        # Print summary
        self._print_evaluation_summary(results)

        return results

    def _print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation summary for a model.

        Args:
            results (Dict[str, Any]): Evaluation results
        """
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS FOR {results['model_name'].upper()}")
        print(f"{'='*50}")

        print(f"Accuracy:     {results['accuracy']:.4f}")
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:       {results['recall']:.4f}")
        print(f"F1-Score:     {results['f1_score']:.4f}")
        print(f"Specificity:  {results['specificity']:.4f}")
        print(f"MCC:          {results['mcc']:.4f}")

        if results['roc_auc'] is not None:
            print(f"ROC-AUC:      {results['roc_auc']:.4f}")
        if results['pr_auc'] is not None:
            print(f"PR-AUC:       {results['pr_auc']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"TN: {results['tn']}, FP: {results['fp']}")
        print(f"FN: {results['fn']}, TP: {results['tp']}")

        print(f"\nFalse Positive Rate: {results['false_positive_rate']:.4f}")
        print(f"False Negative Rate: {results['false_negative_rate']:.4f}")

    def plot_confusion_matrix(self, model_name: str, save_path: str = None) -> None:
        """
        Plot confusion matrix for a specific model.

        Args:
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")

        results = self.evaluation_results[model_name]
        cm = results['confusion_matrix']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraudulent'],
                   yticklabels=['Legitimate', 'Fraudulent'])

        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=10, color='red')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, model_names: List[str] = None, save_path: str = None) -> None:
        """
        Plot ROC curves for one or more models.

        Args:
            model_names (List[str]): List of model names. If None, plots all models.
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())

        plt.figure(figsize=(10, 8))

        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"Warning: Model {model_name} not found in evaluation results")
                continue

            results = self.evaluation_results[model_name]
            if results['y_pred_proba'] is None:
                print(f"Warning: No probability predictions available for {model_name}")
                continue

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            auc_score = results['roc_auc']

            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{model_name} (AUC = {auc_score:.3f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, model_names: List[str] = None, save_path: str = None) -> None:
        """
        Plot Precision-Recall curves for one or more models.

        Args:
            model_names (List[str]): List of model names. If None, plots all models.
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())

        plt.figure(figsize=(10, 8))

        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"Warning: Model {model_name} not found in evaluation results")
                continue

            results = self.evaluation_results[model_name]
            if results['y_pred_proba'] is None:
                print(f"Warning: No probability predictions available for {model_name}")
                continue

            # Calculate Precision-Recall curve
            precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
            pr_auc = results['pr_auc']

            plt.plot(recall, precision, linewidth=2,
                    label=f'{model_name} (AP = {pr_auc:.3f})')

        # Calculate baseline (random classifier performance)
        fraud_rate = sum([results['y_test'].sum() for results in self.evaluation_results.values()]) / \
                    sum([len(results['y_test']) for results in self.evaluation_results.values()])

        plt.axhline(y=fraud_rate, color='k', linestyle='--', linewidth=1,
                   label=f'Random (AP = {fraud_rate:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")

        plt.show()

    def plot_feature_importance(self, model: Any, feature_names: List[str],
                              model_name: str = None, top_n: int = 20,
                              save_path: str = None) -> None:
        """
        Plot feature importance for tree-based models or coefficient-based models.

        Args:
            model: Trained model
            feature_names (List[str]): List of feature names
            model_name (str): Name of the model
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot
        """
        if model_name is None:
            model_name = type(model).__name__

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            title = f'Feature Importance - {model_name}'
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            title = f'Feature Coefficients (Absolute) - {model_name}'
        else:
            print(f"Model {model_name} does not support feature importance visualization")
            return

        # Create DataFrame and sort
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_df, x='importance', y='feature', palette='viridis')
        plt.title(title)
        plt.xlabel('Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

        # Print top features
        print(f"\nTop {min(top_n, len(feature_df))} most important features:")
        for i, row in feature_df.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance metrics across all evaluated models.

        Returns:
            pd.DataFrame: Comparison table of model performances
        """
        if not self.evaluation_results:
            print("No models have been evaluated yet")
            return pd.DataFrame()

        comparison_data = []

        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score'],
                'Specificity': results['specificity'],
                'MCC': results['mcc'],
                'ROC_AUC': results['roc_auc'],
                'PR_AUC': results['pr_auc'],
                'FPR': results['false_positive_rate'],
                'FNR': results['false_negative_rate']
            })

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df

    def plot_model_comparison(self, metrics: List[str] = None, save_path: str = None) -> None:
        """
        Plot comparison of multiple models across different metrics.

        Args:
            metrics (List[str]): List of metrics to compare
            save_path (str): Path to save the plot
        """
        if metrics is None:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']

        comparison_df = self.compare_models()
        if comparison_df.empty:
            return

        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(available_metrics[:6]):
            if i < len(axes):
                ax = axes[i]
                comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")

        plt.show()

    def generate_classification_report(self, model_name: str) -> str:
        """
        Generate detailed classification report for a model.

        Args:
            model_name (str): Name of the model

        Returns:
            str: Classification report
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")

        results = self.evaluation_results[model_name]
        report = classification_report(
            results['y_test'],
            results['y_pred'],
            target_names=['Legitimate', 'Fraudulent'],
            digits=4
        )

        print(f"\nClassification Report for {model_name}:")
        print("=" * 60)
        print(report)

        return report

def evaluate_fraud_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series,
                         generate_plots: bool = True, save_plots: bool = False,
                         plot_save_path: str = 'plots/') -> Dict[str, Any]:
    """
    Convenience function to evaluate multiple fraud detection models.

    Args:
        models (Dict[str, Any]): Dictionary of trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        generate_plots (bool): Whether to generate evaluation plots
        save_plots (bool): Whether to save plots to disk
        plot_save_path (str): Path to save plots

    Returns:
        Dict[str, Any]: Evaluation results and evaluator instance
    """
    evaluator = FraudDetectionEvaluator()

    # Evaluate all models
    all_results = {}
    for model_name, model in models.items():
        results = evaluator.evaluate_model(model, X_test, y_test, model_name)
        all_results[model_name] = results

    # Generate comparison
    comparison_df = evaluator.compare_models()
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False, float_format='%.4f'))

    # Generate plots if requested
    if generate_plots:
        import os
        if save_plots:
            os.makedirs(plot_save_path, exist_ok=True)

        # ROC Curve
        evaluator.plot_roc_curve(
            save_path=os.path.join(plot_save_path, 'roc_curves.png') if save_plots else None
        )

        # Precision-Recall Curve
        evaluator.plot_precision_recall_curve(
            save_path=os.path.join(plot_save_path, 'pr_curves.png') if save_plots else None
        )

        # Model Comparison
        evaluator.plot_model_comparison(
            save_path=os.path.join(plot_save_path, 'model_comparison.png') if save_plots else None
        )

        # Individual confusion matrices
        for model_name in models.keys():
            evaluator.plot_confusion_matrix(
                model_name,
                save_path=os.path.join(plot_save_path, f'{model_name}_confusion_matrix.png') if save_plots else None
            )

    return {
        'evaluator': evaluator,
        'results': all_results,
        'comparison': comparison_df
    }