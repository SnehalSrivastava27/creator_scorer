#!/usr/bin/env python3
"""
Combined prediction script for both credibility and storytelling scoring.
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
    DATA_DIR, INPUT_CSV, RESULTS_DIR,
    CREDIBILITY_FEATURES, STORYTELLING_FEATURES
)

def load_prediction_data(input_path=None):
    """Load data for prediction."""
    print("📂 Loading prediction data...")
    
    if input_path:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path)
    else:
        # Use default input file
        default_path = DATA_DIR / INPUT_CSV
        if not default_path.exists():
            raise FileNotFoundError(f"Default input file not found: {default_path}")
        df = pd.read_csv(default_path)
    
    print(f"   Data shape: {df.shape}")
    
    # Check for required features
    available_cred_features = [f for f in CREDIBILITY_FEATURES if f in df.columns]
    available_story_features = [f for f in STORYTELLING_FEATURES if f in df.columns]
    
    missing_cred_features = [f for f in CREDIBILITY_FEATURES if f not in df.columns]
    missing_story_features = [f for f in STORYTELLING_FEATURES if f not in df.columns]
    
    print(f"   Available credibility features: {len(available_cred_features)}/{len(CREDIBILITY_FEATURES)}")
    print(f"   Available storytelling features: {len(available_story_features)}/{len(STORYTELLING_FEATURES)}")
    
    if missing_cred_features:
        print(f"   ⚠️  Missing credibility features: {missing_cred_features}")
    if missing_story_features:
        print(f"   ⚠️  Missing storytelling features: {missing_story_features}")
    
    if len(available_cred_features) == 0 and len(available_story_features) == 0:
        raise ValueError("No required features found in data")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Predict both credibility and storytelling scores')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Skip confidence interval calculation')
    parser.add_argument('--show-features', action='store_true',
                       help='Show feature importance from trained models')
    parser.add_argument('--credibility-only', action='store_true',
                       help='Predict only credibility scores')
    parser.add_argument('--storytelling-only', action='store_true',
                       help='Predict only storytelling scores')
    
    args = parser.parse_args()
    
    try:
        # Initialize combined scorer
        scorer = CombinedScorer()
        
        # Load models
        scorer.load_models()
        
        # Show feature importance if requested
        if args.show_features:
            print("\n🎯 Feature Importance Comparison:")
            
            if scorer.credibility_scorer.is_fitted:
                print("\n   Top Credibility Features:")
                cred_importance = scorer.credibility_scorer.get_feature_importance(top_n=8)
                for _, row in cred_importance.iterrows():
                    print(f"     {row['feature']}: {row['importance']:.3f}")
            
            if scorer.storytelling_scorer.is_fitted:
                print("\n   Top Storytelling Features:")
                story_importance = scorer.storytelling_scorer.get_feature_importance(top_n=8)
                for _, row in story_importance.iterrows():
                    print(f"     {row['feature']}: {row['importance']:.3f}")
            print()
        
        # Load data
        df = load_prediction_data(args.input)
        
        # Make predictions based on what models are available and what user wants
        if args.credibility_only:
            if not scorer.credibility_scorer.is_fitted:
                raise ValueError("Credibility model not trained. Run train_combined.py first.")
            
            print("🎯 Making credibility predictions only...")
            if args.no_confidence:
                predictions = scorer.credibility_scorer.predict(df)
                results_df = df.copy()
                results_df['credibility_score'] = predictions
            else:
                cred_results = scorer.credibility_scorer.predict_with_confidence(df)
                results_df = df.copy()
                results_df['credibility_score'] = cred_results['predictions']
                results_df['credibility_lower_bound'] = cred_results['lower_bound']
                results_df['credibility_upper_bound'] = cred_results['upper_bound']
                results_df['credibility_uncertainty'] = cred_results['uncertainty']
            
            # Add category
            results_df['credibility_category'] = pd.cut(
                results_df['credibility_score'],
                bins=[0, 3, 6, 8, 10],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
            
        elif args.storytelling_only:
            if not scorer.storytelling_scorer.is_fitted:
                raise ValueError("Storytelling model not trained. Run train_combined.py first.")
            
            print("🎯 Making storytelling predictions only...")
            if args.no_confidence:
                predictions = scorer.storytelling_scorer.predict(df)
                results_df = df.copy()
                results_df['storytelling_score'] = predictions
            else:
                story_results = scorer.storytelling_scorer.predict_with_confidence(df)
                results_df = df.copy()
                results_df['storytelling_score'] = story_results['predictions']
                results_df['storytelling_lower_bound'] = story_results['lower_bound']
                results_df['storytelling_upper_bound'] = story_results['upper_bound']
                results_df['storytelling_uncertainty'] = story_results['uncertainty']
            
            # Add category
            results_df['storytelling_category'] = pd.cut(
                results_df['storytelling_score'],
                bins=[0, 3, 6, 8, 10],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )
            
        else:
            # Combined predictions (default)
            output_path = args.output if args.output else None
            results_df = scorer.save_combined_predictions(df, output_path)
        
        # Save results if not already saved
        if args.credibility_only or args.storytelling_only:
            output_path = args.output if args.output else RESULTS_DIR / "predictions.csv"
            
            # Sort by score (highest first)
            if 'credibility_score' in results_df.columns:
                results_df = results_df.sort_values('credibility_score', ascending=False)
            elif 'storytelling_score' in results_df.columns:
                results_df = results_df.sort_values('storytelling_score', ascending=False)
            
            results_df.to_csv(output_path, index=False)
            print(f"💾 Predictions saved to {output_path}")
        
        # Print summary
        if not args.credibility_only and not args.storytelling_only:
            scorer.print_combined_summary(results_df)
        else:
            print(f"\n📊 Prediction Summary:")
            print(f"   Total creators analyzed: {len(results_df)}")
            
            if 'credibility_score' in results_df.columns:
                print(f"   Credibility scores: {results_df['credibility_score'].min():.2f} - {results_df['credibility_score'].max():.2f}")
                print(f"   Mean credibility: {results_df['credibility_score'].mean():.2f}")
                
                if 'credibility_category' in results_df.columns:
                    category_counts = results_df['credibility_category'].value_counts()
                    print("   Credibility distribution:")
                    for category, count in category_counts.items():
                        percentage = (count / len(results_df)) * 100
                        print(f"     {category}: {count} ({percentage:.1f}%)")
            
            if 'storytelling_score' in results_df.columns:
                print(f"   Storytelling scores: {results_df['storytelling_score'].min():.2f} - {results_df['storytelling_score'].max():.2f}")
                print(f"   Mean storytelling: {results_df['storytelling_score'].mean():.2f}")
                
                if 'storytelling_category' in results_df.columns:
                    category_counts = results_df['storytelling_category'].value_counts()
                    print("   Storytelling distribution:")
                    for category, count in category_counts.items():
                        percentage = (count / len(results_df)) * 100
                        print(f"     {category}: {count} ({percentage:.1f}%)")
        
        print(f"\n✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()