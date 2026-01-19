"""
Basic usage example for the Instagram Reel Feature Extraction System.
"""
from pipeline.feature_extractor import feature_extractor

def main():
    """Example of basic usage."""
    
    # List of Instagram creators to analyze (without @ symbol)
    creators = [
        "username1",
        "username2", 
        "username3"
    ]
    
    print("🎬 Starting feature extraction...")
    
    # Extract features from all creators
    results_df = feature_extractor.process_multiple_creators(
        creators=creators,
        use_parallel=True  # Use parallel processing for faster results
    )
    
    if not results_df.empty:
        # Save results to CSV
        output_file = "example_results.csv"
        feature_extractor.save_results(results_df, output_file)
        
        # Display some basic statistics
        print(f"\n📊 Analysis Summary:")
        print(f"   Total reels analyzed: {len(results_df)}")
        print(f"   Creators processed: {results_df['creator'].nunique()}")
        
        # Show average scores across all reels
        numeric_features = [
            'multi_cue_attr_0_10',
            'eye_contact_score_0_10', 
            'hist_score_0_10',
            'sun_exposure_0_10_A',
            'avg_words_spoken_non_music'
        ]
        
        available_features = [f for f in numeric_features if f in results_df.columns]
        if available_features:
            print(f"\n📈 Average Scores:")
            for feature in available_features:
                avg_score = results_df[feature].mean()
                print(f"   {feature}: {avg_score:.2f}")
        
        print(f"\n💾 Results saved to: {output_file}")
    else:
        print("❌ No features were extracted. Check your configuration and try again.")

if __name__ == "__main__":
    main()