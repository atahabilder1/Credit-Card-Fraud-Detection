"""
Main pipeline for Credit Card Fraud Detection.
Orchestrates the complete machine learning workflow from data preprocessing to model evaluation.
"""

import os
import sys
import argparse
import time
from datetime import datetime
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import preprocess_data
from src.train import train_fraud_detection_models
from src.evaluate import evaluate_fraud_models
from src.utils import ensure_directory_exists, save_model_results

warnings.filterwarnings('ignore')

def main(config=None):
    """
    Main function to run the complete fraud detection pipeline.

    Args:
        config (dict): Configuration parameters for the pipeline
    """
    # Default configuration
    default_config = {
        'data_path': 'data/creditcard_2023.csv',
        'test_size': 0.2,
        'scaler_type': 'standard',
        'sampling_method': 'smote',
        'use_grid_search': True,
        'cv_folds': 5,
        'save_models': True,
        'save_plots': True,
        'models_dir': 'models/',
        'plots_dir': 'plots/',
        'random_state': 42
    }

    # Update with user config if provided
    if config:
        default_config.update(config)

    config = default_config

    print("CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {config}")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1: Data Preprocessing
        print("\n=� STEP 1: DATA PREPROCESSING")
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
        print(f" Preprocessing completed in {preprocessing_time:.2f} seconds")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Step 2: Model Training
        print("\n> STEP 2: MODEL TRAINING")
        print("-" * 40)

        training_start = time.time()

        # Ensure models directory exists
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
        print(f" Training completed in {training_time:.2f} seconds")

        # Step 3: Model Evaluation
        print("\n=� STEP 3: MODEL EVALUATION")
        print("-" * 40)

        evaluation_start = time.time()

        # Ensure plots directory exists if saving plots
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
        print(f" Evaluation completed in {evaluation_time:.2f} seconds")

        # Step 4: Generate Summary Report
        print("\n=� STEP 4: SUMMARY REPORT")
        print("-" * 40)

        generate_summary_report(
            training_comparison,
            evaluation_comparison,
            config,
            preprocessing_time,
            training_time,
            evaluation_time
        )

        # Step 5: Save Results
        if config['save_models']:
            print("\n=� STEP 5: SAVING RESULTS")
            print("-" * 40)

            # Save comprehensive results
            pipeline_results = {
                'config': config,
                'preprocessing_data': {
                    'feature_columns': feature_columns,
                    'scaler': preprocessed_data['scaler'],
                    'preprocessing_params': preprocessed_data['preprocessing_params']
                },
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'training_comparison': training_comparison,
                'evaluation_comparison': evaluation_comparison,
                'execution_time': {
                    'preprocessing': preprocessing_time,
                    'training': training_time,
                    'evaluation': evaluation_time,
                    'total': time.time() - start_time
                }
            }

            results_path = os.path.join(config['models_dir'], 'pipeline_results.pkl')
            save_model_results(pipeline_results, results_path)

            # Save evaluation comparison as CSV
            evaluation_comparison.to_csv(
                os.path.join(config['models_dir'], 'model_comparison.csv'),
                index=False
            )
            print(f"=� Model comparison saved to {config['models_dir']}model_comparison.csv")

        total_time = time.time() - start_time
        print(f"\n<� PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

        return pipeline_results

    except Exception as e:
        print(f"\nL PIPELINE FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_summary_report(training_comparison, evaluation_comparison, config,
                          preprocessing_time, training_time, evaluation_time):
    """
    Generate a comprehensive summary report of the pipeline execution.
    """
    print("=� FRAUD DETECTION PIPELINE SUMMARY")
    print("=" * 60)

    # Configuration summary
    print(f"Dataset: {config['data_path']}")
    print(f"Test size: {config['test_size']}")
    print(f"Scaling method: {config['scaler_type']}")
    print(f"Sampling method: {config['sampling_method']}")
    print(f"Grid search: {'Enabled' if config['use_grid_search'] else 'Disabled'}")
    print(f"Cross-validation folds: {config['cv_folds']}")

    # Timing summary
    total_time = preprocessing_time + training_time + evaluation_time
    print(f"\n�  EXECUTION TIME BREAKDOWN:")
    print(f"Preprocessing: {preprocessing_time:.2f}s ({preprocessing_time/total_time*100:.1f}%)")
    print(f"Training: {training_time:.2f}s ({training_time/total_time*100:.1f}%)")
    print(f"Evaluation: {evaluation_time:.2f}s ({evaluation_time/total_time*100:.1f}%)")
    print(f"Total: {total_time:.2f}s")

    # Training performance summary
    print(f"\n<�  TRAINING PERFORMANCE (Cross-Validation F1 Scores):")
    for _, row in training_comparison.iterrows():
        print(f"{row['Model']}: {row['Mean_F1_Score']:.4f} (�{row['Std_F1_Score']:.4f})")

    # Test performance summary
    print(f"\n<� TEST PERFORMANCE:")
    # Sort by F1 score for better readability
    eval_sorted = evaluation_comparison.sort_values('F1_Score', ascending=False)

    print(f"{'Model':<20} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'ROC-AUC':<8}")
    print("-" * 60)
    for _, row in eval_sorted.iterrows():
        roc_auc = row['ROC_AUC'] if row['ROC_AUC'] is not None else 0.0
        print(f"{row['Model']:<20} {row['Precision']:<10.4f} {row['Recall']:<8.4f} "
              f"{row['F1_Score']:<8.4f} {roc_auc:<8.4f}")

    # Best model recommendation
    best_model = eval_sorted.iloc[0]
    print(f"\n<� RECOMMENDED MODEL: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1_Score']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f}")

    # Key insights
    print(f"\n=� KEY INSIGHTS:")

    fraud_detection_rate = best_model['Recall']
    false_alarm_rate = best_model['FPR']

    print(f"" Best model detects {fraud_detection_rate*100:.1f}% of fraudulent transactions")
    print(f"" False alarm rate: {false_alarm_rate*100:.2f}% of legitimate transactions")
    print(f"" Model precision: {best_model['Precision']*100:.1f}% of fraud alerts are genuine")

    if config['sampling_method'] != 'none':
        print(f"" Resampling technique ({config['sampling_method']}) was applied to handle class imbalance")

    print(f"\n=� OUTPUTS:")
    if config['save_models']:
        print(f"" Trained models saved in: {config['models_dir']}")
    if config['save_plots']:
        print(f"" Evaluation plots saved in: {config['plots_dir']}")

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Pipeline')

    parser.add_argument('--data-path', type=str, default='data/creditcard_2023.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--scaler', type=str, default='standard', choices=['standard', 'robust'],
                       help='Type of feature scaler to use')
    parser.add_argument('--sampling', type=str, default='smote',
                       choices=['smote', 'adasyn', 'under', 'smotetomek', 'none'],
                       help='Sampling method for handling imbalanced data')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Disable grid search hyperparameter tuning')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directory to save trained models')
    parser.add_argument('--plots-dir', type=str, default='plots/',
                       help='Directory to save evaluation plots')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Disable saving trained models')
    parser.add_argument('--no-save-plots', action='store_true',
                       help='Disable saving evaluation plots')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')

    return parser.parse_args()

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        args = parse_arguments()
        config = {
            'data_path': args.data_path,
            'test_size': args.test_size,
            'scaler_type': args.scaler,
            'sampling_method': args.sampling,
            'use_grid_search': not args.no_grid_search,
            'cv_folds': args.cv_folds,
            'save_models': not args.no_save_models,
            'save_plots': not args.no_save_plots,
            'models_dir': args.models_dir,
            'plots_dir': args.plots_dir,
            'random_state': args.random_state
        }
    else:
        config = None  # Use default configuration

    # Run the pipeline
    results = main(config)

    if results is None:
        sys.exit(1)  # Exit with error code if pipeline failed
    else:
        print("\n<� Thank you for using the Credit Card Fraud Detection Pipeline!")
        print("For questions or issues, please check the project documentation.")