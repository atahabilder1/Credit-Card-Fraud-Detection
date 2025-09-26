"""
Unit tests for the preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor(random_state=42)

        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # Generate random features
        data = {
            'id': range(n_samples),
            'Amount': np.random.exponential(scale=100, size=n_samples),
            'Class': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        }

        # Add V features
        for i in range(1, n_features + 1):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)

        self.sample_df = pd.DataFrame(data)

        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary file
        os.unlink(self.temp_file.name)

    def test_load_and_validate_data(self):
        """Test data loading and validation."""
        df = self.preprocessor.load_and_validate_data(self.temp_file.name)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        self.assertIn('Class', df.columns)
        self.assertIn('Amount', df.columns)

    def test_clean_data(self):
        """Test data cleaning."""
        # Add some missing values
        test_df = self.sample_df.copy()
        test_df.loc[0:10, 'V1'] = np.nan

        cleaned_df = self.preprocessor.clean_data(test_df)

        # Check that missing values are handled
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
        self.assertEqual(len(cleaned_df), len(test_df))

    def test_feature_engineering(self):
        """Test feature engineering."""
        engineered_df = self.preprocessor.feature_engineering(self.sample_df)

        # Check that new features are created
        self.assertIn('Amount_log', engineered_df.columns)
        self.assertIn('Amount_bin_encoded', engineered_df.columns)

        # Check that feature columns are stored
        self.assertIsNotNone(self.preprocessor.feature_columns)
        self.assertIsInstance(self.preprocessor.feature_columns, list)

    def test_split_data(self):
        """Test data splitting."""
        preprocessed_df = self.preprocessor.feature_engineering(self.sample_df)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            preprocessed_df, test_size=0.2
        )

        # Check shapes
        total_samples = len(preprocessed_df)
        expected_train_size = int(total_samples * 0.8)

        self.assertAlmostEqual(len(X_train), expected_train_size, delta=5)
        self.assertAlmostEqual(len(X_test), total_samples - len(X_train), delta=5)

        # Check that features and target are separated correctly
        self.assertNotIn('Class', X_train.columns)
        self.assertNotIn('Class', X_test.columns)

if __name__ == '__main__':
    unittest.main()