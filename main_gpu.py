"""
GPU-Optimized Credit Card Fraud Detection Pipeline.
Leverages NVIDIA GPU and multi-core CPU for maximum performance.
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
    GPU-optimized main function for fraud detection pipeline.
    """
    # GPU-optimized configuration
    config = {
        'data_path': 'data/creditcard_2023.csv',
        'test_size': 0.2,
        'scaler_type': 'standard',
        'sampling_method': 'smote',
        'use_grid_search': True,  # Enable hyperparameter tuning
        'cv_folds': 5,
        'save_models': True,
        'save_plots': True,
        'models_dir': 'models/',
        'plots_dir': 'plots/',
        'random_state': 42
    }

    print("CREDIT CARD FRAUD DETECTION PIPELINE - GPU OPTIMIZED")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {config}")
    print("=" * 60)

    # Check GPU availability
    try:
        import cupy as cp
        print(f"GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"GPU Memory: {cp.cuda.runtime.memGetInfo()[1] / 1024**3:.1f} GB")
    except:
        print("GPU acceleration not available, using CPU optimization")

    start_time = time.time()

    try:
        # Step 1: Data Preprocessing
        print("\nSTEP 1: DATA PREPROCESSING")
        print("-" * 40)

        preprocessing_start = time.time()

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
        feature_columns = preprocessed_data['feature_columns']

        preprocessing_time = time.time() - preprocessing_start
        print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Step 2: Model Training (GPU-optimized)
        print("\nSTEP 2: MODEL TRAINING (GPU-OPTIMIZED)")
        print("-" * 40)

        training_start = time.time()
        ensure_directory_exists(config['models_dir'])

        training_results = train_fraud_detection_models(
            X_train, y_train,
            use_grid_search=config['use_grid_search'],
            cv_folds=config['cv_folds'],
            save_models=config['save_models'],
            save_path=config['models_dir']
        )

        trained_models = training_results['trainer'].best_models
        training_comparison = training_results['comparison']

        training_time = time.time() - training_start
        print(f"Training completed in {training_time:.2f} seconds")

        # Step 3: Model Evaluation
        print("\nSTEP 3: MODEL EVALUATION")
        print("-" * 40)

        evaluation_start = time.time()

        if config['save_plots']:
            ensure_directory_exists(config['plots_dir'])

        evaluation_results = evaluate_fraud_models(
            trained_models, X_test, y_test,
            generate_plots=True,
            save_plots=config['save_plots'],
            plot_save_path=config['plots_dir']
        )

        evaluation_comparison = evaluation_results['comparison']

        evaluation_time = time.time() - evaluation_start
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")

        # Step 4: Performance Summary
        print("\nSTEP 4: PERFORMANCE ANALYSIS")
        print("-" * 40)

        total_time = time.time() - start_time

        # Find best model
        best_model_row = evaluation_comparison.loc[evaluation_comparison['F1_Score'].idxmax()]
        best_model_name = best_model_row['Model']

        print("FRAUD DETECTION PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Dataset: {config['data_path']}")
        print(f"Total transactions: {len(X_train) + len(X_test):,}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Features: {len(feature_columns)}")
        print(f"Sampling method: {config['sampling_method']}")

        print(f"\nEXECUTION TIMES:")
        print(f"Preprocessing: {preprocessing_time:.2f}s")
        print(f"Training: {training_time:.2f}s")
        print(f"Evaluation: {evaluation_time:.2f}s")
        print(f"Total: {total_time:.2f}s ({total_time/60:.1f} minutes)")

        print(f"\nMODEL PERFORMANCE (Test Set):")
        print("=" * 50)
        for _, row in evaluation_comparison.iterrows():
            print(f"{row['Model']}:")
            print(f"  Precision: {row['Precision']:.4f}")
            print(f"  Recall: {row['Recall']:.4f}")
            print(f"  F1-Score: {row['F1_Score']:.4f}")
            if row['ROC_AUC'] is not None:
                print(f"  ROC-AUC: {row['ROC_AUC']:.4f}")
            print()

        print(f"BEST MODEL: {best_model_name}")
        print(f"Best F1-Score: {best_model_row['F1_Score']:.4f}")
        print(f"Fraud Detection Rate: {best_model_row['Recall']*100:.1f}%")
        print(f"False Alarm Rate: {best_model_row['FPR']*100:.2f}%")

        # Save final results
        final_results = {
            'config': config,
            'execution_times': {
                'preprocessing': preprocessing_time,
                'training': training_time,
                'evaluation': evaluation_time,
                'total': total_time
            },
            'best_model': {
                'name': best_model_name,
                'metrics': best_model_row.to_dict()
            },
            'all_results': evaluation_comparison.to_dict('records')
        }

        import joblib
        results_path = os.path.join(config['models_dir'], 'final_results.pkl')
        joblib.dump(final_results, results_path)
        print(f"\nResults saved to: {results_path}")

        if config['save_plots']:
            print(f"Visualizations saved to: {config['plots_dir']}")

        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
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