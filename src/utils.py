"""
Utility functions for credit card fraud detection project.
Contains helper functions for data loading, visualization, and common operations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load credit card dataset from CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    return df

def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataset.

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        Dict[str, Any]: Dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'class_distribution': df['Class'].value_counts().to_dict() if 'Class' in df.columns else None,
        'fraud_percentage': (df['Class'].sum() / len(df) * 100) if 'Class' in df.columns else None
    }

    return info

def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset.

    Args:
        df (pd.DataFrame): Dataset to summarize
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    info = get_dataset_info(df)

    print(f"Shape: {info['shape']}")
    print(f"Features: {len(info['columns'])}")
    print(f"Missing values: {sum(info['missing_values'].values())}")

    if info['class_distribution']:
        print(f"\nClass Distribution:")
        for class_val, count in info['class_distribution'].items():
            label = 'Legitimate' if class_val == 0 else 'Fraudulent'
            percentage = (count / info['shape'][0]) * 100
            print(f"  {label} (Class {class_val}): {count:,} ({percentage:.3f}%)")

        print(f"\nImbalance Ratio: {info['class_distribution'][0] / info['class_distribution'][1]:.1f}:1")

    print("=" * 60)

def plot_class_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the distribution of classes in the dataset.

    Args:
        df (pd.DataFrame): Dataset
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Count plot
    plt.subplot(1, 2, 1)
    class_counts = df['Class'].value_counts()
    plt.bar(['Legitimate (0)', 'Fraudulent (1)'], class_counts.values,
            color=['skyblue', 'salmon'])
    plt.title('Class Distribution (Count)')
    plt.ylabel('Number of Transactions')

    # Add count labels on bars
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + max(class_counts.values) * 0.01, f'{v:,}',
                ha='center', va='bottom')

    # Percentage plot
    plt.subplot(1, 2, 2)
    percentages = df['Class'].value_counts(normalize=True) * 100
    plt.pie(percentages.values, labels=['Legitimate (0)', 'Fraudulent (1)'],
            autopct='%1.3f%%', colors=['skyblue', 'salmon'])
    plt.title('Class Distribution (Percentage)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_feature_distributions(df: pd.DataFrame, features: list = None,
                             n_cols: int = 4, figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Plot distributions of selected features by class.

    Args:
        df (pd.DataFrame): Dataset
        features (list, optional): List of features to plot. If None, plots first 16 features.
        n_cols (int): Number of columns in subplot grid
        figsize (tuple): Figure size
    """
    if features is None:
        # Select first 16 V features for visualization
        features = [col for col in df.columns if col.startswith('V')][:16]

    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(features):
        ax = axes[i]

        # Plot distributions for each class
        df[df['Class'] == 0][feature].hist(bins=50, alpha=0.7, label='Legitimate',
                                          ax=ax, color='skyblue', density=True)
        df[df['Class'] == 1][feature].hist(bins=50, alpha=0.7, label='Fraudulent',
                                          ax=ax, color='salmon', density=True)

        ax.set_title(f'{feature} Distribution')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def create_correlation_matrix(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create and plot correlation matrix of features.

    Args:
        df (pd.DataFrame): Dataset
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})

    plt.title('Feature Correlation Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {save_path}")

    plt.show()

def save_model_results(results: Dict[str, Any], file_path: str) -> None:
    """
    Save model results to a file.

    Args:
        results (dict): Dictionary containing model results
        file_path (str): Path to save the results
    """
    import joblib

    joblib.dump(results, file_path)
    print(f"Results saved to {file_path}")

def load_model_results(file_path: str) -> Dict[str, Any]:
    """
    Load model results from a file.

    Args:
        file_path (str): Path to the results file

    Returns:
        Dict[str, Any]: Loaded results
    """
    import joblib

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found at {file_path}")

    results = joblib.load(file_path)
    print(f"Results loaded from {file_path}")

    return results

def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")