"""
Combined scoring system for both credibility and storytelling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from .model import CredibilityScorer
from .storytelling_model import StorytellingScorer
from .config import (
    DATA_DIR, INPUT_CSV, RESULTS_DIR,
    CREDIBILITY_FEATURES, STORYTELLING_FEATURES
)

class CombinedScorer:
    """Combined credibility and storytelling scoring system."""
    
    def __init__(self):
        self.credibility_scorer = CredibilityScorer()
        self.storytelling_scorer = StorytellingScorer()
        self.models_loaded = False
        
    def load_models(self):
        """Load both trained models."""
        print("🔄 Loading trained models...")
        
        try:
            self.credibility_scorer.load()
            print("   ✅ Credibility model loaded")
        except FileNotFoundError:
            print("   ⚠️  Credibility model not found - train it first")
            
        try:
            self.storytelling_scorer.load()
            print("   ✅ Storytelling model loaded")
        except FileNotFoundError:
            print("   ⚠️  Storytelling model not found - train it first")
            
        self.models_loaded = True
    
    def create_sample_labels(self, df, output_path=None):
        """Create sample labels for both credibility and storytelling."""
        print("📝 Creating sample labels for both models...")
        print("   Note: In production, use human-labeled scores!")
        
        df_labeled = df.copy()
        
        # Create credibility scores (same as before)
        credibility_score = 5.0
        
        if 'eye_contact_avg_score_0_10' in df.columns:
            credibility_score += (df['eye_contact_avg_score_0_10'] / 10) * 2
        if 'mean_face_density' in df.columns:
            credibility_score += df['mean_face_density'] * 1.5
        if 'gemini_is_marketing' in df.columns:
            credibility_score -= df['gemini_is_marketing'] * 2
        if 'series_reel_mean' in df.columns:
            credibility_score += df['series_reel_mean'] * 1
        if 'gemini_comment_sentiment_counts.agreeing' in df.columns:
            max_agreeing = df['gemini_comment_sentiment_counts.agreeing'].max()
            if max_agreeing > 0:
                credibility_score += (df['gemini_comment_sentiment_counts.agreeing'] / max_agreeing) * 1.5
        
        # Add noise and clip
        np.random.seed(42)
        noise = np.random.normal(0, 0.5, len(df))
        credibility_score += noise
        df_labeled['credibility_score'] = np.clip(credibility_score, 1, 10)
        
        # Create storytelling scores
        storytelling_score = 5.0
        
        if 'avg_captioned_reels' in df.columns:
            storytelling_score += df['avg_captioned_reels'] * 2  # Captions help storytelling
        if 'gemini_has_humour' in df.columns:
            storytelling_score += df['gemini_has_humour'] * 1.5  # Humor enhances stories
        if 'mean_face_density' in df.columns:
            storytelling_score += df['mean_face_density'] * 1  # Face presence for connection
        if 'outlier_2sigma_ratio' in df.columns:
            storytelling_score += df['outlier_2sigma_ratio'] * 0.5  # Variety in content
        if 'gemini_comment_sentiment_counts.agreeing' in df.columns:
            max_agreeing = df['gemini_comment_sentiment_counts.agreeing'].max()
            if max_agreeing > 0:
                storytelling_score += (df['gemini_comment_sentiment_counts.agreeing'] / max_agreeing) * 1
        if 'gemini_comment_sentiment_counts.neutral' in df.columns:
            # Neutral comments might indicate thoughtful content
            max_neutral = df['gemini_comment_sentiment_counts.neutral'].max()
            if max_neutral > 0:
                storytelling_score += (df['gemini_comment_sentiment_counts.neutral'] / max_neutral) * 0.5
        
        # Add different noise for storytelling and clip
        np.random.seed(43)
        noise_story = np.random.normal(0, 0.6, len(df))
        storytelling_score += noise_story
        df_labeled['storytelling_score'] = np.clip(storytelling_score, 1, 10)
        
        if output_path:
            df_labeled.to_csv(output_path, index=False)
            print(f"   Sample labeled data saved to {output_path}")
        
        return df_labeled
    
    def train_both_models(self, df, test_size=0.2):
        """Train both credibility and storytelling models."""
        print("🚀 Training both models...")
        
        # Check if we have labels, if not create sample ones
        if 'credibility_score' not in df.columns or 'storytelling_score' not in df.columns:
            print("   Creating sample labels...")
            df = self.create_sample_labels(df)
        
        # Train credibility model
        print("\n--- Training Credibility Model ---")
        try:
            cred_results = self.credibility_scorer.train(df, 'credibility_score', test_size)
            self.credibility_scorer.save()
        except Exception as e:
            print(f"   ❌ Credibility training failed: {e}")
            cred_results = None
        
        # Train storytelling model
        print("\n--- Training Storytelling Model ---")
        try:
            story_results = self.storytelling_scorer.train(df, 'storytelling_score', test_size)
            self.storytelling_scorer.save()
        except Exception as e:
            print(f"   ❌ Storytelling training failed: {e}")
            story_results = None
        
        return {
            'credibility_results': cred_results,
            'storytelling_results': story_results
        }
    
    def predict_combined(self, df, with_confidence=True):
        """Make predictions with both models."""
        if not self.models_loaded:
            self.load_models()
        
        print("🎯 Making combined predictions...")
        
        results = {'creator': df.get('creator', range(len(df)))}
        
        # Credibility predictions
        try:
            if self.credibility_scorer.is_fitted:
                if with_confidence:
                    cred_results = self.credibility_scorer.predict_with_confidence(df)
                    results['credibility_score'] = cred_results['predictions']
                    results['credibility_lower_bound'] = cred_results['lower_bound']
                    results['credibility_upper_bound'] = cred_results['upper_bound']
                    results['credibility_uncertainty'] = cred_results['uncertainty']
                else:
                    results['credibility_score'] = self.credibility_scorer.predict(df)
                
                # Add credibility category
                results['credibility_category'] = pd.cut(
                    results['credibility_score'],
                    bins=[0, 3, 6, 8, 10],
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    include_lowest=True
                )
                
                print(f"   ✅ Credibility predictions: {results['credibility_score'].min():.2f} - {results['credibility_score'].max():.2f}")
            else:
                print("   ⚠️  Credibility model not trained")
                results['credibility_score'] = np.full(len(df), 5.0)
                results['credibility_category'] = 'Unknown'
        except Exception as e:
            print(f"   ❌ Credibility prediction failed: {e}")
            results['credibility_score'] = np.full(len(df), 5.0)
            results['credibility_category'] = 'Error'
        
        # Storytelling predictions
        try:
            if self.storytelling_scorer.is_fitted:
                if with_confidence:
                    story_results = self.storytelling_scorer.predict_with_confidence(df)
                    results['storytelling_score'] = story_results['predictions']
                    results['storytelling_lower_bound'] = story_results['lower_bound']
                    results['storytelling_upper_bound'] = story_results['upper_bound']
                    results['storytelling_uncertainty'] = story_results['uncertainty']
                else:
                    results['storytelling_score'] = self.storytelling_scorer.predict(df)
                
                # Add storytelling category
                results['storytelling_category'] = pd.cut(
                    results['storytelling_score'],
                    bins=[0, 3, 6, 8, 10],
                    labels=['Poor', 'Fair', 'Good', 'Excellent'],
                    include_lowest=True
                )
                
                print(f"   ✅ Storytelling predictions: {results['storytelling_score'].min():.2f} - {results['storytelling_score'].max():.2f}")
            else:
                print("   ⚠️  Storytelling model not trained")
                results['storytelling_score'] = np.full(len(df), 5.0)
                results['storytelling_category'] = 'Unknown'
        except Exception as e:
            print(f"   ❌ Storytelling prediction failed: {e}")
            results['storytelling_score'] = np.full(len(df), 5.0)
            results['storytelling_category'] = 'Error'
        
        return pd.DataFrame(results)
    
    def save_combined_predictions(self, df, output_path=None):
        """Make predictions and save to CSV with all original data."""
        if output_path is None:
            output_path = RESULTS_DIR / "combined_predictions.csv"
        
        print("💾 Generating combined predictions...")
        
        # Make predictions
        predictions_df = self.predict_combined(df)
        
        # Combine with original data
        results_df = df.copy()
        
        # Add prediction columns
        for col in predictions_df.columns:
            if col != 'creator':
                results_df[col] = predictions_df[col]
        
        # Create overall score (weighted combination)
        if 'credibility_score' in results_df.columns and 'storytelling_score' in results_df.columns:
            results_df['overall_score'] = (
                results_df['credibility_score'] * 0.6 +  # Credibility weighted more
                results_df['storytelling_score'] * 0.4
            )
            
            results_df['overall_category'] = pd.cut(
                results_df['overall_score'],
                bins=[0, 4, 6, 8, 10],
                labels=['Below Average', 'Average', 'Above Average', 'Exceptional'],
                include_lowest=True
            )
        
        # Sort by overall score (highest first)
        if 'overall_score' in results_df.columns:
            results_df = results_df.sort_values('overall_score', ascending=False)
        elif 'credibility_score' in results_df.columns:
            results_df = results_df.sort_values('credibility_score', ascending=False)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print(f"   ✅ Combined predictions saved to {output_path}")
        
        return results_df
    
    def print_combined_summary(self, results_df):
        """Print summary of combined predictions."""
        print("\n📊 Combined Scoring Summary:")
        print(f"   Total creators analyzed: {len(results_df)}")
        
        if 'credibility_score' in results_df.columns:
            print(f"\n   🎯 Credibility Scores:")
            print(f"     Mean: {results_df['credibility_score'].mean():.2f}")
            print(f"     Range: {results_df['credibility_score'].min():.2f} - {results_df['credibility_score'].max():.2f}")
            
            if 'credibility_category' in results_df.columns:
                cred_counts = results_df['credibility_category'].value_counts()
                for category, count in cred_counts.items():
                    percentage = (count / len(results_df)) * 100
                    print(f"     {category}: {count} ({percentage:.1f}%)")
        
        if 'storytelling_score' in results_df.columns:
            print(f"\n   📚 Storytelling Scores:")
            print(f"     Mean: {results_df['storytelling_score'].mean():.2f}")
            print(f"     Range: {results_df['storytelling_score'].min():.2f} - {results_df['storytelling_score'].max():.2f}")
            
            if 'storytelling_category' in results_df.columns:
                story_counts = results_df['storytelling_category'].value_counts()
                for category, count in story_counts.items():
                    percentage = (count / len(results_df)) * 100
                    print(f"     {category}: {count} ({percentage:.1f}%)")
        
        if 'overall_score' in results_df.columns:
            print(f"\n   🏆 Overall Scores:")
            print(f"     Mean: {results_df['overall_score'].mean():.2f}")
            print(f"     Range: {results_df['overall_score'].min():.2f} - {results_df['overall_score'].max():.2f}")
            
            print(f"\n   🥇 Top 5 Overall Performers:")
            top_5 = results_df.head(5)
            for _, row in top_5.iterrows():
                creator = row.get('creator', 'Unknown')
                overall = row.get('overall_score', 0)
                cred = row.get('credibility_score', 0)
                story = row.get('storytelling_score', 0)
                print(f"     {creator}: {overall:.2f} (Cred: {cred:.2f}, Story: {story:.2f})")
    
    def get_feature_importance_comparison(self):
        """Compare feature importance between models."""
        if not self.models_loaded:
            self.load_models()
        
        comparison = {}
        
        if self.credibility_scorer.is_fitted:
            cred_importance = self.credibility_scorer.get_feature_importance()
            comparison['credibility'] = cred_importance
        
        if self.storytelling_scorer.is_fitted:
            story_importance = self.storytelling_scorer.get_feature_importance()
            comparison['storytelling'] = story_importance
        
        return comparison