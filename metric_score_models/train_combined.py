#!/usr/bin/env python3
"""
Training script for both credibility and storytelling scoring models.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent))

from metric_score_models.combined_scorer import CombinedScorer
from metric_score_models.config import (
    DATA_DIR, INPUT_CSV, LABELED_DATA_CSV, RESULTS_DIR,
    CREDIBILITY_FEATURES, STORYTELLING_FEATURES
)

def load_data(input_path=None, use_sample_labels=False):
    """Load training data."""
    print("📂 Loading data for combined training...")
    
    # Try to load existing labeled data first
    labeled_path = DATA_DIR / LABELED_DATA_CSV
    
    if input_path:
        print(f"   Loading data from {input_path}")
        df = pd.read_csv(input_path)
    elif labeled_path.exists() and not use_sample_labels:
        print(f"   Loading labeled data from {labeled_path}")
        df = pd.read_csv(labeled_path)
    else:
        # Use default input file
        input_path = DATA_DIR / INPUT_CSV
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        print(f"   Loading feature data from {input_path}")
        df = pd.read_csv(input_path)
    
    print(f"   Data shape: {df.shape}")
    
    # Check available features
    available_cred_features = [f for f in CREDIBILITY_FEATURES if f in df.columns]
    available_story_features = [f for f in STORYTELLING_FEATURES if f in df.columns]
    
    print(f"   Available credibility features: {len(available_cred_features)}/{len(CREDIBILITY_FEATURES)}")
    print(f"   Available storytelling features: {len(available_story_features)}/{len(STORYTELLING_FEATURES)}")
    
    missing_cred = [f for f in CREDIBILITY_FEATURES if f not in df.columns]
    missing_story = [f for f in STORYTELLING_FEATURES if f not in df.columns]
    
    if missing_cred:
        print(f"   ⚠️  Missing credibility features: {missing_cred}")
    if missing_story:
        print(f"   ⚠️  Missing storytelling features: {missing_story}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Train both credibility and storytelling models')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--sample-labels', action='store_true', 
                       help='Create sample labels (for demo purposes)')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Perform hyperparameter tuning for both models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--credibility-only', action='store_true',
                       help='Train only credibility model')
    parser.add_argument('--storytelling-only', action='store_true',
                       help='Train only storytelling model')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_data(args.input, args.sample_labels)
        
        # Initialize combined scorer
        scorer = CombinedScorer()
        
        # Hyperparameter tuning if requested
        if args.tune_hyperparams:
            print("🔧 Performing hyperparameter tuning...")
            
            # Create sample labels if needed
            if 'credibility_score' not in df.columns or 'storytelling_score' not in df.columns:
                df = scorer.create_sample_labels(df)
            
            if not args.storytelling_only:
                print("   Tuning credibility model...")
                cred_best_params = scorer.credibility_scorer.hyperparameter_tuning(df, 'credibility_score')
                print(f"   Best credibility parameters: {cred_best_params}")
            
            if not args.credibility_only:
                print("   Tuning storytelling model...")
                story_best_params = scorer.storytelling_scorer.hyperparameter_tuning(df, 'storytelling_score')
                print(f"   Best storytelling parameters: {story_best_params}")
        
        # Train models
        if args.credibility_only:
            print("🚀 Training credibility model only...")
            if 'credibility_score' not in df.columns:
                df = scorer.create_sample_labels(df)
            results = scorer.credibility_scorer.train(df, 'credibility_score', args.test_size)
            scorer.credibility_scorer.save()
            
        elif args.storytelling_only:
            print("🚀 Training storytelling model only...")
            if 'storytelling_score' not in df.columns:
                df = scorer.create_sample_labels(df)
            results = scorer.storytelling_scorer.train(df, 'storytelling_score', args.test_size)
            scorer.storytelling_scorer.save()
            
        else:
            print("🚀 Training both models...")
            results = scorer.train_both_models(df, args.test_size)
        
        # Generate and save feature importance plots
        plots_dir = RESULTS_DIR / "training_plots"
        plots_dir.mkdir(exist_ok=True)
        
        if not args.storytelling_only and scorer.credibility_scorer.is_fitted:
            cred_plot_path = plots_dir / "credibility_feature_importance.png"
            scorer.credibility_scorer.plot_feature_importance(save_path=cred_plot_path)
        
        if not args.credibility_only and scorer.storytelling_scorer.is_fitted:
            story_plot_path = plots_dir / "storytelling_feature_importance.png"
            scorer.storytelling_scorer.plot_feature_importance(save_path=story_plot_path)
        
        # Print feature importance comparison
        if not args.credibility_only and not args.storytelling_only:
            print("\n🎯 Feature Importance Comparison:")
            
            if scorer.credibility_scorer.is_fitted:
                print("\n   Top Credibility Features:")
                cred_importance = scorer.credibility_scorer.get_feature_importance(top_n=5)
                for _, row in cred_importance.iterrows():
                    print(f"     {row['feature']}: {row['importance']:.3f}")
            
            if scorer.storytelling_scorer.is_fitted:
                print("\n   Top Storytelling Features:")
                story_importance = scorer.storytelling_scorer.get_feature_importance(top_n=5)
                for _, row in story_importance.iterrows():
                    print(f"     {row['feature']}: {row['importance']:.3f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Models saved to: metric_score_models/trained_models/")
        print(f"   Use predict_combined.py to make predictions on new data")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()