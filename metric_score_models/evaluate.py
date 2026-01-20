#!/usr/bin/env python3
"""
Model evaluation and analysis script.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from metric_score_models.model import CredibilityScorer
from metric_score_models.config import DATA_DIR, LABELED_DATA_CSV, RESULTS_DIR

def load_evaluation_data(input_path=None):
    """Load data with ground truth labels for evaluation."""
    print("📂 Loading evaluation data...")
    
    if input_path:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path)
    else:
        # Use default labeled data
        default_path = DATA_DIR / LABELED_DATA_CSV
        if not default_path.exists():
            raise FileNotFoundError(f"Labeled data not found: {default_path}")
        df = pd.read_csv(default_path)
    
    if 'credibility_score' not in df.columns:
        raise ValueError("Evaluation data must contain 'credibility_score' column")
    
    print(f"   Data shape: {df.shape}")
    return df

def evaluate_model(scorer, df):
    """Evaluate model performance."""
    print("📊 Evaluating model performance...")
    
    # Make predictions
    predictions = scorer.predict(df)
    actual = df['credibility_score'].values
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    
    # Correlation
    correlation = np.corrcoef(actual, predictions)[0, 1]
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'correlation': correlation,
        'n_samples': len(actual)
    }
    
    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE: {mae:.3f}")
    print(f"   R²: {r2:.3f}")
    print(f"   MAPE: {mape:.1f}%")
    print(f"   Correlation: {correlation:.3f}")
    
    return metrics, predictions, actual

def plot_predictions_vs_actual(predictions, actual, save_path=None):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(actual, predictions, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Formatting
    plt.xlabel('Actual Credibility Score')
    plt.ylabel('Predicted Credibility Score')
    plt.title('Predicted vs Actual Credibility Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add correlation text
    correlation = np.corrcoef(actual, predictions)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")
    
    plt.show()

def plot_residuals(predictions, actual, save_path=None):
    """Plot residuals analysis."""
    residuals = actual - predictions
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(predictions, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Actual
    axes[1, 1].scatter(actual, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Actual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to {save_path}")
    
    plt.show()

def analyze_feature_impact(scorer, df, save_path=None):
    """Analyze feature impact on predictions."""
    print("🔍 Analyzing feature impact...")
    
    # Get feature importance
    importance_df = scorer.get_feature_importance()
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance for Credibility Scoring')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()
    
    return importance_df

def analyze_prediction_distribution(predictions, actual, save_path=None):
    """Analyze distribution of predictions."""
    plt.figure(figsize=(12, 6))
    
    # Distribution comparison
    plt.subplot(1, 2, 1)
    plt.hist(actual, bins=20, alpha=0.7, label='Actual', color='blue', edgecolor='black')
    plt.hist(predictions, bins=20, alpha=0.7, label='Predicted', color='red', edgecolor='black')
    plt.xlabel('Credibility Score')
    plt.ylabel('Frequency')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot comparison
    plt.subplot(1, 2, 2)
    data_to_plot = [actual, predictions]
    labels = ['Actual', 'Predicted']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Credibility Score')
    plt.title('Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate credibility scoring model')
    parser.add_argument('--input', type=str, help='Input CSV file with ground truth labels')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print("🔄 Loading trained model...")
        scorer = CredibilityScorer()
        scorer.load()
        print("   ✅ Model loaded successfully")
        
        # Load evaluation data
        df = load_evaluation_data(args.input)
        
        # Evaluate model
        metrics, predictions, actual = evaluate_model(scorer, df)
        
        # Generate plots
        if args.save_plots:
            # Create plots directory
            plots_dir = RESULTS_DIR / "evaluation_plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_predictions_vs_actual(predictions, actual, 
                                     plots_dir / "predictions_vs_actual.png")
            plot_residuals(predictions, actual, 
                          plots_dir / "residuals_analysis.png")
            analyze_feature_impact(scorer, df, 
                                  plots_dir / "feature_importance.png")
            analyze_prediction_distribution(predictions, actual,
                                          plots_dir / "distribution_comparison.png")
        else:
            plot_predictions_vs_actual(predictions, actual)
            plot_residuals(predictions, actual)
            analyze_feature_impact(scorer, df)
            analyze_prediction_distribution(predictions, actual)
        
        # Save detailed evaluation results
        results_df = pd.DataFrame({
            'creator': df['creator'] if 'creator' in df.columns else range(len(df)),
            'actual_credibility': actual,
            'predicted_credibility': predictions,
            'absolute_error': np.abs(actual - predictions),
            'squared_error': (actual - predictions) ** 2
        })
        
        results_path = RESULTS_DIR / "evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"📊 Detailed evaluation results saved to {results_path}")
        
        # Save metrics summary
        metrics_df = pd.DataFrame([metrics])
        metrics_path = RESULTS_DIR / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"📈 Evaluation metrics saved to {metrics_path}")
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()