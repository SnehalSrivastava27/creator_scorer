"""
Main feature extraction pipeline.
Orchestrates all feature extraction modules to analyze Instagram reels.
"""
import os
import pandas as pd
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from data_sources.apify_client import instagram_scraper
from features.attractiveness import attractiveness_analyzer
from features.eye_contact import eye_contact_analyzer
from features.creativity import creativity_analyzer
from features.sequential import sequential_analyzer
from features.video_captions import video_caption_analyzer
from features.accessories import accessory_analyzer
from features.sun_exposure import sun_exposure_analyzer
from features.transcript import transcript_analyzer
from features.english_detection import english_detection_analyzer
from features.gemini_analysis import gemini_analyzer
from utils.cache_manager import download_reel_cached
from config import MAX_REELS_PER_CREATOR, MAX_DOWNLOAD_WORKERS

class FeatureExtractor:
    """Main feature extraction pipeline for Instagram reels."""
    
    def __init__(self):
        self.scraper = instagram_scraper
        self.max_reels_per_creator = MAX_REELS_PER_CREATOR
        self.max_workers = MAX_DOWNLOAD_WORKERS
    
    def extract_features_for_reel(self, reel_data: Dict[str, Any], video_path: str) -> Dict[str, Any]:
        """
        Extract all features for a single reel.
        
        Args:
            reel_data: Dict containing reel metadata (url, caption, comments)
            video_path: Path to the downloaded video file
            
        Returns:
            Dict with all extracted features
        """
        print(f"  📊 Extracting features for reel: {reel_data.get('reel_url', 'Unknown')}")
        
        features = {
            "reel_url": reel_data.get("reel_url", ""),
            "caption": reel_data.get("caption", ""),
            "comments": reel_data.get("flat_comments", []),
        }
        
        # Generate transcript first (needed by multiple analyzers)
        print("    🎤 Generating transcript...")
        transcript_metrics = transcript_analyzer.compute_transcript_metrics_for_reel(video_path)
        transcript = transcript_metrics.get("transcript", "")
        features.update(transcript_metrics)
        
        # Visual analysis features
        print("    👁️ Analyzing attractiveness...")
        attractiveness_metrics = attractiveness_analyzer.compute_attractiveness_for_reel(video_path)
        features.update(attractiveness_metrics)
        
        print("    👀 Analyzing eye contact...")
        eye_contact_metrics = eye_contact_analyzer.compute_eye_contact_for_reel(video_path)
        features.update(eye_contact_metrics)
        
        print("    🎨 Analyzing creativity...")
        creativity_metrics = creativity_analyzer.compute_creativity_for_reel(video_path)
        features.update(creativity_metrics)
        
        print("    ☀️ Analyzing sun exposure...")
        sun_exposure_metrics = sun_exposure_analyzer.compute_sun_exposure_for_reel(video_path)
        features.update(sun_exposure_metrics)
        
        print("    📝 Analyzing video captions...")
        caption_metrics = video_caption_analyzer.compute_video_caption_flag_for_reel(video_path)
        features.update(caption_metrics)
        
        print("    👜 Analyzing accessories...")
        accessory_metrics = accessory_analyzer.compute_accessories_for_reel(video_path)
        features.update(accessory_metrics)
        
        # Text-based analysis features
        print("    📺 Analyzing sequential content...")
        sequential_metrics = sequential_analyzer.detect_series_from_text(
            features["caption"], transcript
        )
        features.update(sequential_metrics)
        
        print("    🇺🇸 Analyzing English content...")
        english_metrics = english_detection_analyzer.compute_english_metrics_for_reel(transcript)
        features.update(english_metrics)
        
        print("    🤖 Analyzing with Gemini AI...")
        gemini_metrics = gemini_analyzer.compute_gemini_metrics_for_reel(
            features["caption"], transcript, features["comments"]
        )
        features.update(gemini_metrics)
        
        print("  ✅ Feature extraction completed")
        return features
    
    def process_creator_reels(self, creator: str) -> pd.DataFrame:
        """
        Process all reels for a single creator.
        
        Args:
            creator: Instagram handle (without @)
            
        Returns:
            DataFrame with extracted features for all reels
        """
        print(f"\n🎬 Processing creator: @{creator}")
        
        # Fetch reels from Apify
        reels_df = self.scraper.load_or_fetch_reels_cached(creator, self.max_reels_per_creator)
        
        if reels_df.empty:
            print(f"  ❌ No reels found for @{creator}")
            return pd.DataFrame()
        
        print(f"  📱 Found {len(reels_df)} reels for @{creator}")
        
        all_features = []
        
        for idx, reel_row in reels_df.iterrows():
            reel_data = reel_row.to_dict()
            reel_url = reel_data.get("reel_url", "")
            
            print(f"  🎥 Processing reel {idx + 1}/{len(reels_df)}: {reel_url}")
            
            # Download reel (placeholder - you need to implement actual download)
            video_path = download_reel_cached(reel_url, idx, f"creator_{creator}")
            
            if not video_path or not os.path.exists(video_path):
                print(f"    ❌ Failed to download reel: {reel_url}")
                continue
            
            # Extract features
            try:
                features = self.extract_features_for_reel(reel_data, video_path)
                features["creator"] = creator
                features["reel_index"] = idx
                all_features.append(features)
                
            except Exception as e:
                print(f"    ❌ Error extracting features: {e}")
                continue
        
        if not all_features:
            print(f"  ❌ No features extracted for @{creator}")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(all_features)
        print(f"  ✅ Extracted features for {len(features_df)} reels from @{creator}")
        
        return features_df
    
    def process_multiple_creators(self, creators: List[str], use_parallel: bool = True) -> pd.DataFrame:
        """
        Process multiple creators and extract features from their reels.
        
        Args:
            creators: List of Instagram handles (without @)
            use_parallel: Whether to use parallel processing
            
        Returns:
            Combined DataFrame with features from all creators
        """
        print(f"\n🚀 Starting feature extraction for {len(creators)} creators")
        print(f"📊 Max reels per creator: {self.max_reels_per_creator}")
        print(f"⚡ Parallel processing: {use_parallel}")
        
        all_dataframes = []
        
        if use_parallel and len(creators) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(len(creators), 4)) as executor:
                future_to_creator = {
                    executor.submit(self.process_creator_reels, creator): creator 
                    for creator in creators
                }
                
                for future in as_completed(future_to_creator):
                    creator = future_to_creator[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_dataframes.append(df)
                    except Exception as e:
                        print(f"❌ Error processing @{creator}: {e}")
        else:
            # Sequential processing
            for creator in creators:
                try:
                    df = self.process_creator_reels(creator)
                    if not df.empty:
                        all_dataframes.append(df)
                except Exception as e:
                    print(f"❌ Error processing @{creator}: {e}")
        
        if not all_dataframes:
            print("❌ No features extracted from any creator")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"\n✅ Feature extraction completed!")
        print(f"📊 Total reels processed: {len(combined_df)}")
        print(f"👥 Creators processed: {combined_df['creator'].nunique()}")
        print(f"📈 Features extracted: {len(combined_df.columns)}")
        
        return combined_df
    
    def save_results(self, df: pd.DataFrame, output_path: str = "extracted_features.csv"):
        """
        Save extracted features to a CSV file.
        
        Args:
            df: DataFrame with extracted features
            output_path: Path to save the CSV file
        """
        if df.empty:
            print("❌ No data to save")
            return
        
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
        
        # Print summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   Total reels: {len(df)}")
        print(f"   Unique creators: {df['creator'].nunique()}")
        print(f"   Features per reel: {len(df.columns)}")
        
        # Show sample of numeric features
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 Sample numeric features:")
            print(df[numeric_cols].describe().round(2))

# Global feature extractor instance
feature_extractor = FeatureExtractor()