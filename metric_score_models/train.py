#!/usr/bin/env python3
"""
Training script for credibility scoring model.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent))

from metric_score_models.model import CredibilityScorer
from metric_score_models.config import (
    DATA_DIR, INPUT_CSV, LABELED_DATA_CSV, RESULTS_DIR,
    CREDIBILITY_FEATURES
)

def create_sample_labels(df, output_path=None):
    """
    Create sample credibility labels for demonstration.
    In practice, you would have human-labeled data.
    """
    print("📝 Creating sample credibility labels...")
    print("   Note: In production, use human-labeled credibility scores!")
    
    # Create synthetic credibility scores based on feature combinations
    # This is just for demonstration - replace with real labels
    df_labeled = df.copy()
    
    # Normalize features to 0-1 scale for scoring
    features_to_use = [f for f in CREDIBILITY_FEATURES if f in df.columns]
    
    if not features_to_use:
        raise ValueError(f"No credibility features found in data. Available: {list(df.columns)}")
    
    # Simple heuristic scoring (replace with real labels)
    credibility_score = 5.0  # Base score
    
    # Eye contact contributes positively
    if 'eye_contact_avg_score_0_10' in df.columns:
        credibility_score += (df['eye_contact_avg_score_0_10'] / 10) * 2
    
    # Face density contributes positively
    if 'mean_face_density' in df.columns:
        credibility_score += df['mean_face_density'] * 1.5
    
    # Marketing content reduces credibility
    if 'gemini_is_marketing' in df.columns:
        credibility_score -= df['gemini_is_marketing'] * 2
    
    # Series content increases credibility (shows planning)
    if 'series_reel_mean' in df.columns:
        credibility_score += df['series_reel_mean'] * 1
    
    # Agreeing comments increase credibility
    if 'gemini_comment_sentiment_counts.agreeing' in df.columns:
        # Normalize by max value
        max_agreeing = df['gemini_comment_sentiment_counts.agreeing'].max()
        if max_agreeing > 0:
            credibility_score += (df['gemini_comment_sentiment_counts.agreeing'] / max_agreeing) * 1.5
    
    # Add some noise and clip to 1-10 range
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, len(df))
    credibility_score += noise
    credibility_score = np.clip(credibility_score, 1, 10)
    
    df_labeled['credibility_score'] = credibility_score
    
    if output_path:
        df_labeled.to_csv(output_path, index=False)
        print(f"   Sample labeled data saved to {output_path}")
    
    return df_labeled

def load_data(use_sample_labels=False):
    """Load training data."""
    print("📂 Loading data...")
    
    # Try to load existing labeled data first
    labeled_path = DATA_DIR / LABELED_DATA_CSV
    input_path = DATA_DIR / INPUT_CSV
    
    if labeled_path.exists() and not use_sample_labels:
        print(f"   Loading labeled data from {labeled_path}")
        df = pd.read_csv(labeled_path)
        
        if 'credibility_score' not in df.columns:
            raise ValueError("Labeled data must contain 'credibility_score' column")
            
    elif input_path.exists():
        print(f"   Loading feature data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Create sample labels
        df = create_sample_labels(df, labeled_path)
        
    else:
        raise FileNotFoundError(f"Neither {labeled_path} nor {input_path} found")
    
    print(f"   Data shape: {df.shape}")
    print(f"   Available features: {[f for f in CREDIBILITY_FEATURES if f in df.columns]}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Train credibility scoring model')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--sample-labels', action='store_true', 
                       help='Create sample labels (for demo purposes)')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        if args.input:
            df = pd.read_csv(args.input)
            if 'credibility_score' not in df.columns:
                df = create_sample_labels(df)
        else:
            df = load_data(use_sample_labels=args.sample_labels)
        
        # Initialize model
        scorer = CredibilityScorer()
        
        # Hyperparameter tuning if requested
        if args.tune_hyperparams:
            print("🔧 Performing hyperparameter tuning...")
            best_params = scorer.hyperparameter_tuning(df)
            print(f"Best parameters found: {best_params}")
        
        # Train model
        results = scorer.train(df, test_size=args.test_size)
        
        # Save model
        scorer.save()
        
        # Generate and save feature importance plot
        importance_plot_path = RESULTS_DIR / "feature_importance.png"
        scorer.plot_feature_importance(save_path=importance_plot_path)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'creator': df.iloc[results['y_test'].index]['creator'] if 'creator' in df.columns else range(len(results['y_test'])),
            'actual_credibility': results['y_test'],
            'predicted_credibility': results['test_predictions'],
            'error': results['y_test'] - results['test_predictions']
        })
        
        results_path = RESULTS_DIR / "training_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"📊 Detailed results saved to {results_path}")
        
        # Print feature importance
        print("\n🎯 Top 10 Most Important Features:")
        importance_df = scorer.get_feature_importance(top_n=10)
        for _, row in importance_df.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Model saved to: {scorer.model}")
        print(f"   Use predict.py to make predictions on new data")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()