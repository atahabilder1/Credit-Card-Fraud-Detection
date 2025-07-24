"""
Unit tests for the utils module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    load_data, get_dataset_info, print_dataset_summary,
    save_model_results, load_model_results, ensure_directory_exists
)

class TestUtilsFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        data = {
            'Amount': np.random.exponential(scale=100, size=n_samples),
            'Class': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        }

        # Add V features
        for i in range(1, 6):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)

        self.sample_df = pd.DataFrame(data)

        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_data(self):
        """Test data loading function."""
        df = load_data(self.temp_file.name)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn('Class', df.columns)
        self.assertIn('Amount', df.columns)

    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

    def test_get_dataset_info(self):
        """Test dataset info extraction."""
        info = get_dataset_info(self.sample_df)

        self.assertIsInstance(info, dict)
        self.assertIn('shape', info)
        self.assertIn('columns', info)
        self.assertIn('class_distribution', info)
        self.assertIn('fraud_percentage', info)

        self.assertEqual(info['shape'], (100, 7))
        self.assertIsInstance(info['class_distribution'], dict)
        self.assertIsNotNone(info['fraud_percentage'])

    def test_save_and_load_model_results(self):
        """Test saving and loading model results."""
        test_results = {
            'model_name': 'test_model',
            'accuracy': 0.95,
            'precision': 0.90,
            'test_data': [1, 2, 3, 4, 5]
        }

        results_path = os.path.join(self.temp_dir, 'test_results.pkl')

        # Test saving
        save_model_results(test_results, results_path)
        self.assertTrue(os.path.exists(results_path))

        # Test loading
        loaded_results = load_model_results(results_path)
        self.assertEqual(loaded_results, test_results)

    def test_load_model_results_file_not_found(self):
        """Test loading non-existent results file."""
        with self.assertRaises(FileNotFoundError):
            load_model_results('non_existent_results.pkl')

    def test_ensure_directory_exists(self):
        """Test directory creation."""
        new_dir = os.path.join(self.temp_dir, 'new_test_dir')

        # Directory should not exist initially
        self.assertFalse(os.path.exists(new_dir))

        # Create directory
        ensure_directory_exists(new_dir)

        # Directory should now exist
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))

    def test_ensure_directory_exists_already_exists(self):
        """Test directory creation when directory already exists."""
        # This should not raise an error
        ensure_directory_exists(self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

if __name__ == '__main__':
    # Set matplotlib backend to prevent display issues in testing
    plt.switch_backend('Agg')
    unittest.main()