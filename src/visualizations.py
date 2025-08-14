"""
Advanced visualization functions for credit card fraud detection analysis.
Provides specialized plotting functions for model interpretation and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
from typing import Dict, Any, List, Tuple, Optional

warnings.filterwarnings('ignore')

def plot_learning_curves(model, X_train, y_train, model_name: str = None,
                        cv_folds: int = 5, save_path: str = None):
    """
    Plot learning curves to analyze model performance vs training size.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        cv_folds: Number of CV folds
        save_path: Path to save the plot
    """
    from sklearn.model_selection import learning_curve

    if model_name is None:
        model_name = type(model).__name__

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv_folds, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
    )

    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

    plt.plot(train_sizes, val_mean, 'o-', label='Cross-Validation Score', color='red')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")

    plt.show()

def plot_decision_boundary_2d(model, X, y, feature_names: List[str] = None,
                             model_name: str = None, save_path: str = None):
    """
    Plot 2D decision boundary using the two most important features.

    Args:
        model: Trained model
        X: Features
        y: Labels
        feature_names: List of feature names
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if model_name is None:
        model_name = type(model).__name__

    # Select two most important features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        # Use first two features if no importance available
        importances = [1, 1] + [0] * (X.shape[1] - 2)

    # Get indices of two most important features
    top_features = np.argsort(importances)[-2:]
    X_subset = X.iloc[:, top_features]

    if feature_names:
        feature_x, feature_y = feature_names[top_features[0]], feature_names[top_features[1]]
    else:
        feature_x, feature_y = f'Feature_{top_features[0]}', f'Feature_{top_features[1]}'

    # Create mesh
    h = 0.02
    x_min, x_max = X_subset.iloc[:, 0].min() - 1, X_subset.iloc[:, 0].max() + 1
    y_min, y_max = X_subset.iloc[:, 1].min() - 1, X_subset.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create a dummy dataset for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    # Fill other features with median values
    median_values = X.median().values
    full_mesh = np.zeros((mesh_points.shape[0], X.shape[1]))
    full_mesh[:, top_features] = mesh_points

    for i in range(X.shape[1]):
        if i not in top_features:
            full_mesh[:, i] = median_values[i]

    # Predict
    Z = model.predict(full_mesh)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

    # Plot data points
    scatter = plt.scatter(X_subset.iloc[:, 0], X_subset.iloc[:, 1], c=y,
                         cmap=plt.cm.RdYlBu, edgecolors='black', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Decision Boundary - {model_name}')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision boundary plot saved to {save_path}")

    plt.show()

def plot_prediction_distribution(y_pred_proba, y_test, model_name: str = None,
                                save_path: str = None):
    """
    Plot distribution of prediction probabilities for each class.

    Args:
        y_pred_proba: Prediction probabilities
        y_test: True labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if model_name is None:
        model_name = "Model"

    legitimate_probs = y_pred_proba[y_test == 0]
    fraudulent_probs = y_pred_proba[y_test == 1]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(legitimate_probs, bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    plt.hist(fraudulent_probs, bins=50, alpha=0.7, label='Fraudulent', color='red', density=True)
    plt.xlabel('Predicted Probability of Fraud')
    plt.ylabel('Density')
    plt.title(f'Prediction Probability Distribution - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Box plot
    data_to_plot = [legitimate_probs, fraudulent_probs]
    plt.boxplot(data_to_plot, labels=['Legitimate', 'Fraudulent'])
    plt.ylabel('Predicted Probability of Fraud')
    plt.title(f'Prediction Probability Box Plot - {model_name}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution plot saved to {save_path}")

    plt.show()

def plot_threshold_analysis(y_pred_proba, y_test, model_name: str = None,
                           save_path: str = None):
    """
    Plot threshold analysis showing precision, recall, and F1 score vs threshold.

    Args:
        y_pred_proba: Prediction probabilities
        y_test: True labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    if model_name is None:
        model_name = "Model"

    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='red')
    plt.plot(thresholds, f1_scores, label='F1 Score', color='green')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold vs Metrics - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    # Find optimal threshold based on F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(thresholds, f1_scores, color='green', linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--',
               label=f'Optimal Threshold: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score vs Threshold - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    # Precision-Recall curve with threshold
    plt.plot(recalls, precisions, color='purple', linewidth=2)
    plt.scatter(recalls[optimal_idx], precisions[optimal_idx], color='red', s=100,
               label=f'Optimal Point (F1={f1_scores[optimal_idx]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    # Number of predictions vs threshold
    n_positive_preds = [np.sum(y_pred_proba >= t) for t in thresholds]
    plt.plot(thresholds, n_positive_preds, color='orange', linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Positive Predictions')
    plt.title(f'Positive Predictions vs Threshold - {model_name}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold analysis plot saved to {save_path}")

    plt.show()

    return optimal_threshold

def plot_dimensionality_reduction(X, y, method: str = 'tsne', sample_size: int = 5000,
                                 save_path: str = None):
    """
    Plot 2D visualization of high-dimensional data using dimensionality reduction.

    Args:
        X: Features
        y: Labels
        method: Reduction method ('tsne' or 'pca')
        sample_size: Number of samples to plot
        save_path: Path to save the plot
    """
    # Sample data for visualization if too large
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
    else:
        X_sample = X
        y_sample = y

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = 't-SNE Visualization'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization'
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    print(f"Performing {method.upper()} dimensionality reduction...")
    X_reduced = reducer.fit_transform(X_sample)

    plt.figure(figsize=(12, 8))

    # Create scatter plot
    colors = ['blue', 'red']
    labels = ['Legitimate', 'Fraudulent']

    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = y_sample == i
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                   c=color, alpha=0.6, label=label, s=20)

    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'{title} of Credit Card Transactions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if method.lower() == 'pca':
        # Add explained variance information
        explained_variance = reducer.explained_variance_ratio_
        plt.text(0.02, 0.98, f'Explained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{method.upper()} plot saved to {save_path}")

    plt.show()

def plot_model_comparison_radar(comparison_df: pd.DataFrame, save_path: str = None):
    """
    Create a radar chart comparing multiple models across different metrics.

    Args:
        comparison_df: DataFrame with model comparison results
        save_path: Path to save the plot
    """
    # Select metrics for radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity']
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    if len(available_metrics) < 3:
        print("Not enough metrics available for radar chart")
        return

    # Normalize metrics to 0-1 scale for radar chart
    df_normalized = comparison_df.copy()
    for metric in available_metrics:
        df_normalized[metric] = (df_normalized[metric] - df_normalized[metric].min()) / \
                               (df_normalized[metric].max() - df_normalized[metric].min())

    # Number of variables
    N = len(available_metrics)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Colors for different models
    colors = plt.cm.Set1(np.linspace(0, 1, len(comparison_df)))

    for idx, (_, row) in enumerate(df_normalized.iterrows()):
        values = [row[metric] for metric in available_metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics)
    ax.set_ylim(0, 1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Comparison Radar Chart', size=16, y=1.1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Radar chart saved to {save_path}")

    plt.show()

def create_interactive_dashboard(evaluation_results: Dict[str, Any], save_path: str = None):
    """
    Create an interactive dashboard using Plotly.

    Args:
        evaluation_results: Dictionary containing evaluation results
        save_path: Path to save the HTML dashboard
    """
    models = list(evaluation_results.keys())

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ROC Curves', 'Precision-Recall Curves',
                       'Model Comparison', 'Prediction Distributions'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )

    colors = px.colors.qualitative.Set1[:len(models)]

    # ROC Curves
    for i, (model_name, results) in enumerate(evaluation_results.items()):
        if results['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={results["roc_auc"]:.3f})',
                          line=dict(color=colors[i])),
                row=1, col=1
            )

    # Add diagonal line for ROC
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                  line=dict(dash='dash', color='gray')),
        row=1, col=1
    )

    # Precision-Recall Curves
    for i, (model_name, results) in enumerate(evaluation_results.items()):
        if results['y_pred_proba'] is not None:
            precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
            fig.add_trace(
                go.Scatter(x=recall, y=precision, mode='lines', name=f'{model_name} (AP={results["pr_auc"]:.3f})',
                          line=dict(color=colors[i]), showlegend=False),
                row=1, col=2
            )

    # Model Comparison Bar Chart
    metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics:
        y_values = [evaluation_results[model][metric] for model in models
                   if evaluation_results[model][metric] is not None]
        x_values = [model for model in models
                   if evaluation_results[model][metric] is not None]

        fig.add_trace(
            go.Bar(x=x_values, y=y_values, name=metric.replace('_', ' ').title(),
                  showlegend=False),
            row=2, col=1
        )

    # Prediction Distribution (using first model as example)
    first_model = list(evaluation_results.keys())[0]
    first_results = evaluation_results[first_model]

    if first_results['y_pred_proba'] is not None:
        legitimate_probs = first_results['y_pred_proba'][first_results['y_test'] == 0]
        fraudulent_probs = first_results['y_pred_proba'][first_results['y_test'] == 1]

        fig.add_trace(
            go.Histogram(x=legitimate_probs, name='Legitimate', opacity=0.7,
                        showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=fraudulent_probs, name='Fraudulent', opacity=0.7,
                        showlegend=False),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        title='Credit Card Fraud Detection Dashboard',
        height=800,
        showlegend=True
    )

    # Update axes labels
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_xaxes(title_text="Models", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_xaxes(title_text="Prediction Probability", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")

    fig.show()

def plot_cost_benefit_analysis(y_test, y_pred_proba, cost_matrix: Dict[str, float],
                              model_name: str = None, save_path: str = None):
    """
    Plot cost-benefit analysis for different threshold values.

    Args:
        y_test: True labels
        y_pred_proba: Prediction probabilities
        cost_matrix: Dictionary with costs {'tp': 0, 'tn': 0, 'fp': cost, 'fn': cost}
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if model_name is None:
        model_name = "Model"

    thresholds = np.linspace(0, 1, 101)
    total_costs = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate confusion matrix elements
        tp = np.sum((y_test == 1) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))

        # Calculate total cost
        total_cost = (tp * cost_matrix.get('tp', 0) +
                     tn * cost_matrix.get('tn', 0) +
                     fp * cost_matrix.get('fp', 100) +
                     fn * cost_matrix.get('fn', 500))

        total_costs.append(total_cost)

    # Find optimal threshold
    optimal_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = total_costs[optimal_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, total_costs, linewidth=2, color='blue')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--',
               label=f'Optimal Threshold: {optimal_threshold:.3f}')
    plt.scatter(optimal_threshold, min_cost, color='red', s=100, zorder=5)

    plt.xlabel('Threshold')
    plt.ylabel('Total Cost')
    plt.title(f'Cost-Benefit Analysis - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add cost information
    plt.text(0.02, 0.98, f'Min Cost: ${min_cost:.0f}\nOptimal Threshold: {optimal_threshold:.3f}',
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost-benefit analysis plot saved to {save_path}")

    plt.show()

    return optimal_threshold, min_cost