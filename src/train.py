"""
Model training module for credit card fraud detection.
Implements various machine learning algorithms with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings
from typing import Dict, Any, Tuple, Optional

warnings.filterwarnings('ignore')

class FraudDetectionTrainer:
    """
    Class for training fraud detection models.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the trainer.

        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.cv_scores = {}

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize different machine learning models.

        Returns:
            Dict[str, Any]: Dictionary of initialized models
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='gpu_hist',
                gpu_id=0,
                n_jobs=-1
            )
        }

        self.models = models
        print(f"Initialized {len(models)} models: {list(models.keys())}")
        return models

    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for different models.

        Returns:
            Dict[str, Dict]: Hyperparameter grids for each model
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }

        return param_grids

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   use_grid_search: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a specific model with optional hyperparameter tuning.

        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
            cv_folds (int): Number of cross-validation folds

        Returns:
            Dict[str, Any]: Training results including model and scores
        """
        print(f"Training {model_name}...")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        model = self.models[model_name]

        if use_grid_search:
            print(f"Performing hyperparameter tuning for {model_name}...")
            param_grids = self.get_hyperparameter_grids()

            if model_name in param_grids:
                # Use StratifiedKFold for imbalanced datasets
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

                grid_search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=cv,
                    scoring='f1',  # Use F1 score for imbalanced data
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_

                print(f"Best parameters for {model_name}: {best_params}")
                print(f"Best CV F1 score: {best_score:.4f}")

            else:
                print(f"No hyperparameter grid found for {model_name}, using default parameters")
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}
                best_score = None

        else:
            print(f"Training {model_name} with default parameters...")
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}
            best_score = None

        # Perform cross-validation on the best model
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1')

        # Store results
        self.best_models[model_name] = best_model
        self.cv_scores[model_name] = cv_scores

        results = {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        print(f"Cross-validation F1 scores for {model_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Training completed for {model_name}\n")

        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        use_grid_search: bool = True, cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Train all initialized models.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
            cv_folds (int): Number of cross-validation folds

        Returns:
            Dict[str, Dict[str, Any]]: Training results for all models
        """
        print("Starting training for all models...")
        print("=" * 60)

        if not self.models:
            self.initialize_models()

        all_results = {}

        for model_name in self.models.keys():
            try:
                results = self.train_model(
                    model_name, X_train, y_train, use_grid_search, cv_folds
                )
                all_results[model_name] = results

            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue

        # Print summary of all models
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        for model_name, results in all_results.items():
            print(f"{model_name.replace('_', ' ').title()}:")
            print(f"  CV F1 Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
            if results['best_score']:
                print(f"  Grid Search Best Score: {results['best_score']:.4f}")
            print()

        return all_results

    def get_feature_importance(self, model_name: str, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.

        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names

        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found in trained models")

        model = self.best_models[model_name]

        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For logistic regression, use absolute values of coefficients
            importances = np.abs(model.coef_[0])
        else:
            print(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_models(self, save_path: str, models_to_save: list = None) -> None:
        """
        Save trained models to disk.

        Args:
            save_path (str): Base path to save models
            models_to_save (list): List of model names to save. If None, saves all models.
        """
        if models_to_save is None:
            models_to_save = list(self.best_models.keys())

        import os
        os.makedirs(save_path, exist_ok=True)

        for model_name in models_to_save:
            if model_name in self.best_models:
                model_path = os.path.join(save_path, f'{model_name}_model.pkl')
                joblib.dump(self.best_models[model_name], model_path)
                print(f"Saved {model_name} to {model_path}")

        # Save training results
        results_path = os.path.join(save_path, 'training_results.pkl')
        training_data = {
            'cv_scores': self.cv_scores,
            'models': list(self.best_models.keys())
        }
        joblib.dump(training_data, results_path)
        print(f"Saved training results to {results_path}")

    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the saved model

        Returns:
            Any: Loaded model
        """
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        return model

    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.

        Returns:
            pd.DataFrame: Comparison of model performances
        """
        if not self.cv_scores:
            print("No models have been trained yet")
            return pd.DataFrame()

        comparison_data = []

        for model_name, scores in self.cv_scores.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Mean_F1_Score': scores.mean(),
                'Std_F1_Score': scores.std(),
                'Min_F1_Score': scores.min(),
                'Max_F1_Score': scores.max()
            })

        comparison_df = pd.DataFrame(comparison_data).sort_values('Mean_F1_Score', ascending=False)
        return comparison_df

def train_fraud_detection_models(X_train: pd.DataFrame, y_train: pd.Series,
                               use_grid_search: bool = True, cv_folds: int = 5,
                               save_models: bool = True, save_path: str = 'models/') -> Dict[str, Any]:
    """
    Convenience function to train fraud detection models.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        cv_folds (int): Number of cross-validation folds
        save_models (bool): Whether to save trained models
        save_path (str): Path to save models

    Returns:
        Dict[str, Any]: Training results and trainer instance
    """
    trainer = FraudDetectionTrainer()
    trainer.initialize_models()

    # Train all models
    results = trainer.train_all_models(X_train, y_train, use_grid_search, cv_folds)

    # Save models if requested
    if save_models:
        trainer.save_models(save_path)

    # Get model comparison
    comparison = trainer.compare_models()
    print("Model Comparison:")
    print(comparison.to_string(index=False))

    return {
        'trainer': trainer,
        'results': results,
        'comparison': comparison
    }