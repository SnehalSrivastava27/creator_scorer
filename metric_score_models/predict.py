#!/usr/bin/env python3
"""
Prediction script for credibility scoring model.
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
    DATA_DIR, INPUT_CSV, PREDICTIONS_CSV, RESULTS_DIR,
    CREDIBILITY_FEATURES
)

def load_model():
    """Load the trained model."""
    print("🔄 Loading trained model...")
    
    scorer = CredibilityScorer()
    try:
        scorer.load()
        print("   ✅ Model loaded successfully")
        return scorer
    except FileNotFoundError as e:
        print(f"   ❌ Model not found: {e}")
        print("   Please run train.py first to train the model")
        sys.exit(1)

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
    available_features = [f for f in CREDIBILITY_FEATURES if f in df.columns]
    missing_features = [f for f in CREDIBILITY_FEATURES if f not in df.columns]
    
    print(f"   Available features: {len(available_features)}/{len(CREDIBILITY_FEATURES)}")
    
    if missing_features:
        print(f"   ⚠️  Missing features: {missing_features}")
        if len(available_features) == 0:
            raise ValueError("No credibility features found in data")
    
    return df

def make_predictions(scorer, df, with_confidence=True):
    """Make credibility predictions."""
    print("🎯 Making credibility predictions...")
    
    try:
        if with_confidence:
            results = scorer.predict_with_confidence(df)
            predictions = results['predictions']
            lower_bound = results['lower_bound']
            upper_bound = results['upper_bound']
            uncertainty = results['uncertainty']
        else:
            predictions = scorer.predict(df)
            lower_bound = None
            upper_bound = None
            uncertainty = None
        
        print(f"   ✅ Generated {len(predictions)} predictions")
        print(f"   Score range: {predictions.min():.2f} - {predictions.max():.2f}")
        print(f"   Mean score: {predictions.mean():.2f} ± {predictions.std():.2f}")
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': uncertainty
        }
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        raise

def save_predictions(df, prediction_results, output_path=None):
    """Save predictions to CSV."""
    print("💾 Saving predictions...")
    
    if output_path is None:
        output_path = RESULTS_DIR / PREDICTIONS_CSV
    
    # Create results dataframe
    results_df = df.copy()
    results_df['credibility_score'] = prediction_results['predictions']
    
    if prediction_results['lower_bound'] is not None:
        results_df['credibility_lower_bound'] = prediction_results['lower_bound']
        results_df['credibility_upper_bound'] = prediction_results['upper_bound']
        results_df['credibility_uncertainty'] = prediction_results['uncertainty']
    
    # Add credibility category
    results_df['credibility_category'] = pd.cut(
        results_df['credibility_score'],
        bins=[0, 3, 6, 8, 10],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    # Sort by credibility score (highest first)
    results_df = results_df.sort_values('credibility_score', ascending=False)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"   ✅ Predictions saved to {output_path}")
    
    return results_df

def print_summary(results_df):
    """Print prediction summary."""
    print("\n📊 Prediction Summary:")
    print(f"   Total creators analyzed: {len(results_df)}")
    
    # Category distribution
    category_counts = results_df['credibility_category'].value_counts()
    print("\n   Credibility Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"     {category}: {count} ({percentage:.1f}%)")
    
    # Top and bottom performers
    print(f"\n   🏆 Top 5 Most Credible Creators:")
    top_5 = results_df.head(5)
    for _, row in top_5.iterrows():
        creator = row.get('creator', 'Unknown')
        score = row['credibility_score']
        print(f"     {creator}: {score:.2f}")
    
    print(f"\n   ⚠️  Bottom 5 Least Credible Creators:")
    bottom_5 = results_df.tail(5)
    for _, row in bottom_5.iterrows():
        creator = row.get('creator', 'Unknown')
        score = row['credibility_score']
        print(f"     {creator}: {score:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Predict credibility scores')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Skip confidence interval calculation')
    parser.add_argument('--show-features', action='store_true',
                       help='Show feature importance from trained model')
    
    args = parser.parse_args()
    
    try:
        # Load trained model
        scorer = load_model()
        
        # Show feature importance if requested
        if args.show_features:
            print("\n🎯 Feature Importance (from trained model):")
            importance_df = scorer.get_feature_importance(top_n=10)
            for _, row in importance_df.iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
            print()
        
        # Load data
        df = load_prediction_data(args.input)
        
        # Make predictions
        prediction_results = make_predictions(
            scorer, df, 
            with_confidence=not args.no_confidence
        )
        
        # Save results
        results_df = save_predictions(df, prediction_results, args.output)
        
        # Print summary
        print_summary(results_df)
        
        print(f"\n✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()