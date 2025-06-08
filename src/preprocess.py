"""
Data preprocessing module for credit card fraud detection.
Handles data loading, cleaning, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Tuple, Optional, Dict, Any
import warnings

try:
    from .utils import load_data, print_dataset_summary
except ImportError:
    from utils import load_data, print_dataset_summary

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Class for preprocessing credit card fraud detection data.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor.

        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'Class'

    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate the dataset.

        Args:
            file_path (str): Path to the dataset

        Returns:
            pd.DataFrame: Loaded and validated dataset
        """
        # Load data
        df = load_data(file_path)

        # Validate required columns
        required_columns = ['Class', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for V features
        v_features = [col for col in df.columns if col.startswith('V')]
        if len(v_features) == 0:
            raise ValueError("No V features found in dataset")

        print_dataset_summary(df)

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.

        Args:
            df (pd.DataFrame): Raw dataset

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("Cleaning data...")

        # Create a copy to avoid modifying original data
        df_clean = df.copy()

        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            print(f"Found {missing_before} missing values")
            # For numerical features, fill with median
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = df_clean[numeric_columns].fillna(
                df_clean[numeric_columns].median()
            )
            print("Missing values filled with median")

        # Remove duplicates if any
        duplicates_before = df_clean.duplicated().sum()
        if duplicates_before > 0:
            df_clean = df_clean.drop_duplicates()
            print(f"Removed {duplicates_before} duplicate rows")

        # Handle potential data type issues
        df_clean[self.target_column] = df_clean[self.target_column].astype(int)

        print(f"Data cleaning completed. Final shape: {df_clean.shape}")

        return df_clean

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.

        Args:
            df (pd.DataFrame): Cleaned dataset

        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        print("Performing feature engineering...")

        df_features = df.copy()

        # Log transform for Amount (add 1 to handle zero values)
        df_features['Amount_log'] = np.log1p(df_features['Amount'])

        # Create Amount bins
        df_features['Amount_bin'] = pd.cut(df_features['Amount'],
                                         bins=[0, 10, 100, 1000, float('inf')],
                                         labels=['low', 'medium', 'high', 'very_high'])

        # Convert categorical features to numerical
        df_features['Amount_bin_encoded'] = df_features['Amount_bin'].cat.codes

        # Remove the original categorical column
        df_features = df_features.drop('Amount_bin', axis=1)

        # Store feature columns (excluding target and id if present)
        self.feature_columns = [col for col in df_features.columns
                              if col not in [self.target_column, 'id']]

        print(f"Feature engineering completed. Total features: {len(self.feature_columns)}")

        return df_features

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified scaler.

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            scaler_type (str): Type of scaler ('standard' or 'robust')

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test features
        """
        print(f"Scaling features using {scaler_type} scaler...")

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")

        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        # Transform test data using fitted scaler
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        print("Feature scaling completed")

        return X_train_scaled, X_test_scaled

    def handle_imbalanced_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                              method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced dataset using various sampling techniques.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            method (str): Sampling method ('smote', 'adasyn', 'under', 'smotetomek', 'none')

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled training data
        """
        print(f"Handling imbalanced data using {method} method...")

        if method == 'none':
            print("No resampling applied")
            return X_train, y_train

        # Print original class distribution
        original_dist = y_train.value_counts()
        print(f"Original distribution - Legitimate: {original_dist[0]}, Fraudulent: {original_dist[1]}")

        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif method == 'under':
            sampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError("method must be one of: 'smote', 'adasyn', 'under', 'smotetomek', 'none'")

        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name=y_train.name)

        # Print new class distribution
        new_dist = y_resampled.value_counts()
        print(f"New distribution - Legitimate: {new_dist[0]}, Fraudulent: {new_dist[1]}")

        return X_resampled, y_resampled

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2,
                  stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Args:
            df (pd.DataFrame): Preprocessed dataset
            test_size (float): Proportion of data for testing
            stratify (bool): Whether to stratify split by target variable

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        print(f"Splitting data with test_size={test_size}")

        # Separate features and target
        X = df[self.feature_columns]
        y = df[self.target_column]

        # Split data
        stratify_target = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_target
        )

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Print class distribution in splits
        train_dist = y_train.value_counts()
        test_dist = y_test.value_counts()
        print(f"Training - Legitimate: {train_dist[0]}, Fraudulent: {train_dist[1]}")
        print(f"Test - Legitimate: {test_dist[0]}, Fraudulent: {test_dist[1]}")

        return X_train, X_test, y_train, y_test

    def preprocess_pipeline(self, file_path: str, test_size: float = 0.2,
                          scaler_type: str = 'standard', sampling_method: str = 'smote',
                          stratify: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.

        Args:
            file_path (str): Path to the dataset
            test_size (float): Proportion of data for testing
            scaler_type (str): Type of scaler ('standard' or 'robust')
            sampling_method (str): Sampling method for imbalanced data
            stratify (bool): Whether to stratify split by target variable

        Returns:
            Dict[str, Any]: Dictionary containing processed data and metadata
        """
        print("Starting preprocessing pipeline...")

        # Step 1: Load and validate data
        df = self.load_and_validate_data(file_path)

        # Step 2: Clean data
        df_clean = self.clean_data(df)

        # Step 3: Feature engineering
        df_features = self.feature_engineering(df_clean)

        # Step 4: Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df_features, test_size=test_size, stratify=stratify
        )

        # Step 5: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(
            X_train, X_test, scaler_type=scaler_type
        )

        # Step 6: Handle imbalanced data (only on training set)
        X_train_resampled, y_train_resampled = self.handle_imbalanced_data(
            X_train_scaled, y_train, method=sampling_method
        )

        # Prepare results
        results = {
            'X_train': X_train_resampled,
            'X_test': X_test_scaled,
            'y_train': y_train_resampled,
            'y_test': y_test,
            'X_train_original': X_train_scaled,
            'y_train_original': y_train,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'preprocessing_params': {
                'test_size': test_size,
                'scaler_type': scaler_type,
                'sampling_method': sampling_method,
                'stratify': stratify,
                'random_state': self.random_state
            }
        }

        print("Preprocessing pipeline completed successfully!")

        return results

def preprocess_data(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to preprocess credit card fraud data.

    Args:
        file_path (str): Path to the dataset
        **kwargs: Additional preprocessing parameters

    Returns:
        Dict[str, Any]: Dictionary containing processed data and metadata
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_pipeline(file_path, **kwargs)