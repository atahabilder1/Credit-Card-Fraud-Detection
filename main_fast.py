"""
Fast Credit Card Fraud Detection Pipeline for quick results.
Uses optimized settings for faster execution while maintaining quality.
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
    Fast execution pipeline for immediate results.
    """
    # Fast execution configuration
    config = {
        'data_path': 'data/creditcard_2023.csv',
        'test_size': 0.2,
        'scaler_type': 'standard',
        'sampling_method': 'smote',
        'use_grid_search': False,  # Skip hyperparameter tuning for speed
        'cv_folds': 3,             # Reduced CV folds
        'save_models': True,
        'save_plots': True,
        'models_dir': 'models/',
        'plots_dir': 'plots/',
        'random_state': 42
    }

    print("CREDIT CARD FRAUD DETECTION - FAST EXECUTION")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Configuration: Fast mode (no hyperparameter tuning)")
    print("=" * 50)

    # Sample size for faster execution
    sample_size = 50000  # Use 50k samples for speed

    start_time = time.time()

    try:
        # Step 1: Data Preprocessing
        print("\nSTEP 1: DATA PREPROCESSING")
        print("-" * 40)

        preprocessing_start = time.time()

        # Load and sample data for faster execution
        import pandas as pd
        print(f"Loading dataset from {config['data_path']}...")
        df_full = pd.read_csv(config['data_path'])
        print(f"Full dataset size: {df_full.shape}")

        # Stratified sampling to maintain class balance
        if len(df_full) > sample_size:
            df_sample = df_full.groupby('Class', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size//2), random_state=42)
            ).reset_index(drop=True)
            print(f"Using sample of {len(df_sample)} transactions for fast execution")

            # Save sample for preprocessing
            sample_path = 'data/sample_balanced.csv'
            df_sample.to_csv(sample_path, index=False)
            config['data_path'] = sample_path

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

        # Step 2: Model Training
        print("\nSTEP 2: MODEL TRAINING (FAST MODE)")
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

        # Step 4: Results Analysis
        print("\nSTEP 4: RESULTS ANALYSIS")
        print("-" * 40)

        total_time = time.time() - start_time

        # Find best model
        best_model_row = evaluation_comparison.loc[evaluation_comparison['F1_Score'].idxmax()]
        best_model_name = best_model_row['Model']

        print("FRAUD DETECTION RESULTS SUMMARY")
        print("=" * 50)
        print(f"Dataset: {len(df_full):,} total transactions")
        print(f"Sample used: {len(X_train) + len(X_test):,} transactions")
        print(f"Features: {len(feature_columns)}")
        print(f"Execution time: {total_time:.1f} seconds")

        print(f"\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 50)
        for _, row in evaluation_comparison.sort_values('F1_Score', ascending=False).iterrows():
            print(f"{row['Model']}:")
            print(f"  Precision: {row['Precision']:.3f} | Recall: {row['Recall']:.3f} | F1: {row['F1_Score']:.3f}")
            if row['ROC_AUC'] is not None:
                print(f"  ROC-AUC: {row['ROC_AUC']:.3f} | Accuracy: {row['Accuracy']:.3f}")
            print()

        print(f"BEST MODEL: {best_model_name}")
        print(f"✓ F1-Score: {best_model_row['F1_Score']:.3f}")
        print(f"✓ Fraud Detection Rate: {best_model_row['Recall']*100:.1f}%")
        print(f"✓ Precision: {best_model_row['Precision']*100:.1f}%")
        print(f"✓ False Alarm Rate: {best_model_row['FPR']*100:.2f}%")

        # Save results
        results_summary = {
            'execution_time': total_time,
            'best_model': best_model_name,
            'best_metrics': best_model_row.to_dict(),
            'all_models': evaluation_comparison.to_dict('records'),
            'dataset_info': {
                'total_samples': len(df_full),
                'sample_used': len(X_train) + len(X_test),
                'features': len(feature_columns)
            }
        }

        import joblib
        joblib.dump(results_summary, 'models/fast_results.pkl')

        print(f"\n📊 Results saved to: models/fast_results.pkl")
        print(f"📈 Plots saved to: {config['plots_dir']}")
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")

        return True

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)