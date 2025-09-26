"""
Simple main pipeline for Credit Card Fraud Detection.
"""

import os
import sys
import time
from datetime import datetime
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import preprocess_data
from src.train import train_fraud_detection_models
from src.evaluate import evaluate_fraud_models
from src.utils import ensure_directory_exists

warnings.filterwarnings('ignore')

def main():
    """
    Simple main function to test the fraud detection pipeline.
    """
    config = {
        'data_path': 'data/sample_data.csv',
        'test_size': 0.3,
        'scaler_type': 'standard',
        'sampling_method': 'none',
        'use_grid_search': False,
        'cv_folds': 2,
        'save_models': True,
        'save_plots': False,
        'models_dir': 'models/',
        'plots_dir': 'plots/',
        'random_state': 42
    }

    print("CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    start_time = time.time()

    try:
        # Step 1: Data Preprocessing
        print("\nSTEP 1: DATA PREPROCESSING")
        print("-" * 30)

        preprocessed_data = preprocess_data(
            config['data_path'],
            test_size=config['test_size'],
            scaler_type=config['scaler_type'],
            sampling_method=config['sampling_method'],
            stratify=True
        )

        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']
        y_train = preprocessed_data['y_train']
        y_test = preprocessed_data['y_test']

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Step 2: Model Training
        print("\nSTEP 2: MODEL TRAINING")
        print("-" * 30)

        ensure_directory_exists(config['models_dir'])

        training_results = train_fraud_detection_models(
            X_train, y_train,
            use_grid_search=config['use_grid_search'],
            cv_folds=config['cv_folds'],
            save_models=config['save_models'],
            save_path=config['models_dir']
        )

        trained_models = training_results['trainer'].best_models

        # Step 3: Model Evaluation
        print("\nSTEP 3: MODEL EVALUATION")
        print("-" * 30)

        evaluation_results = evaluate_fraud_models(
            trained_models, X_test, y_test,
            generate_plots=False,
            save_plots=False
        )

        total_time = time.time() - start_time
        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {total_time:.2f} seconds")

        return True

    except Exception as e:
        print(f"\nPIPELINE FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)