"""
Example showing how to create custom analysis workflows.
"""
import pandas as pd
from typing import List, Dict, Any

from data_sources.apify_client import instagram_scraper
from features.attractiveness import attractiveness_analyzer
from features.gemini_analysis import gemini_analyzer
from features.transcript import transcript_analyzer

class CustomAnalyzer:
    """Custom analyzer focusing on specific features."""
    
    def __init__(self):
        self.scraper = instagram_scraper
    
    def analyze_creator_attractiveness(self, creator: str) -> Dict[str, Any]:
        """
        Focus only on attractiveness analysis for a creator.
        
        Args:
            creator: Instagram username
            
        Returns:
            Dict with attractiveness metrics
        """
        print(f"🎬 Analyzing attractiveness for @{creator}")
        
        # Get reels
        reels_df = self.scraper.load_or_fetch_reels_cached(creator, max_items=5)
        
        if reels_df.empty:
            return {"error": f"No reels found for @{creator}"}
        
        attractiveness_scores = []
        
        for idx, reel_row in reels_df.iterrows():
            reel_url = reel_row["reel_url"]
            print(f"  📱 Processing reel {idx + 1}: {reel_url}")
            
            # Note: You need to implement actual video download
            # video_path = download_reel(reel_url)
            video_path = None  # Placeholder
            
            if video_path:
                results = attractiveness_analyzer.compute_attractiveness_for_reel(video_path)
                attractiveness_scores.append(results.get('multi_cue_attr_0_10', 0))
        
        if attractiveness_scores:
            return {
                "creator": creator,
                "num_reels": len(attractiveness_scores),
                "avg_attractiveness": sum(attractiveness_scores) / len(attractiveness_scores),
                "max_attractiveness": max(attractiveness_scores),
                "min_attractiveness": min(attractiveness_scores),
                "individual_scores": attractiveness_scores
            }
        else:
            return {"error": "No attractiveness scores computed"}
    
    def analyze_content_themes(self, creators: List[str]) -> pd.DataFrame:
        """
        Analyze content themes across multiple creators using Gemini AI.
        
        Args:
            creators: List of Instagram usernames
            
        Returns:
            DataFrame with content theme analysis
        """
        print(f"🤖 Analyzing content themes for {len(creators)} creators")
        
        results = []
        
        for creator in creators:
            print(f"  👤 Processing @{creator}")
            
            # Get reels
            reels_df = self.scraper.load_or_fetch_reels_cached(creator, max_items=3)
            
            if reels_df.empty:
                continue
            
            creator_themes = {
                "creator": creator,
                "total_reels": len(reels_df),
                "marketing_count": 0,
                "educational_count": 0,
                "vlog_count": 0,
                "humor_count": 0,
                "avg_genz_words": 0
            }
            
            genz_word_counts = []
            
            for idx, reel_row in reels_df.iterrows():
                caption = reel_row.get("caption", "")
                comments = reel_row.get("flat_comments", [])
                
                # For this example, we'll skip transcript generation
                # In practice, you'd generate transcripts here
                transcript = ""
                
                # Analyze with Gemini
                gemini_results = gemini_analyzer.compute_gemini_metrics_for_reel(
                    caption, transcript, comments
                )
                
                # Accumulate theme counts
                creator_themes["marketing_count"] += gemini_results.get("gemini_is_marketing", 0)
                creator_themes["educational_count"] += gemini_results.get("gemini_is_educational", 0)
                creator_themes["vlog_count"] += gemini_results.get("gemini_is_vlog", 0)
                creator_themes["humor_count"] += gemini_results.get("gemini_has_humour", 0)
                
                genz_count = gemini_results.get("gemini_genz_word_count", 0)
                genz_word_counts.append(genz_count)
            
            # Calculate percentages and averages
            if creator_themes["total_reels"] > 0:
                creator_themes["marketing_pct"] = creator_themes["marketing_count"] / creator_themes["total_reels"] * 100
                creator_themes["educational_pct"] = creator_themes["educational_count"] / creator_themes["total_reels"] * 100
                creator_themes["vlog_pct"] = creator_themes["vlog_count"] / creator_themes["total_reels"] * 100
                creator_themes["humor_pct"] = creator_themes["humor_count"] / creator_themes["total_reels"] * 100
                creator_themes["avg_genz_words"] = sum(genz_word_counts) / len(genz_word_counts) if genz_word_counts else 0
            
            results.append(creator_themes)
        
        return pd.DataFrame(results)
    
    def compare_creators(self, creators: List[str]) -> Dict[str, Any]:
        """
        Compare multiple creators across key metrics.
        
        Args:
            creators: List of Instagram usernames
            
        Returns:
            Dict with comparison results
        """
        print(f"⚖️ Comparing {len(creators)} creators")
        
        comparison_data = []
        
        for creator in creators:
            print(f"  📊 Analyzing @{creator}")
            
            # Get basic attractiveness analysis
            attractiveness_data = self.analyze_creator_attractiveness(creator)
            
            if "error" not in attractiveness_data:
                comparison_data.append({
                    "creator": creator,
                    "avg_attractiveness": attractiveness_data.get("avg_attractiveness", 0),
                    "num_reels_analyzed": attractiveness_data.get("num_reels", 0)
                })
        
        if not comparison_data:
            return {"error": "No data available for comparison"}
        
        # Find top performer
        top_creator = max(comparison_data, key=lambda x: x["avg_attractiveness"])
        
        # Calculate overall statistics
        avg_scores = [data["avg_attractiveness"] for data in comparison_data]
        
        return {
            "creators_compared": len(comparison_data),
            "top_performer": top_creator,
            "overall_avg_attractiveness": sum(avg_scores) / len(avg_scores),
            "detailed_results": comparison_data
        }

def main():
    """Example of custom analysis workflows."""
    
    analyzer = CustomAnalyzer()
    
    # Example 1: Attractiveness analysis for a single creator
    print("Example 1: Single Creator Attractiveness Analysis")
    print("=" * 50)
    
    attractiveness_results = analyzer.analyze_creator_attractiveness("example_creator")
    print(f"Results: {attractiveness_results}")
    
    # Example 2: Content theme analysis
    print("\n\nExample 2: Content Theme Analysis")
    print("=" * 50)
    
    creators_for_themes = ["creator1", "creator2", "creator3"]
    theme_df = analyzer.analyze_content_themes(creators_for_themes)
    
    if not theme_df.empty:
        print("Content Theme Results:")
        print(theme_df.to_string(index=False))
    
    # Example 3: Creator comparison
    print("\n\nExample 3: Creator Comparison")
    print("=" * 50)
    
    creators_for_comparison = ["creator1", "creator2"]
    comparison_results = analyzer.compare_creators(creators_for_comparison)
    print(f"Comparison Results: {comparison_results}")

if __name__ == "__main__":
    main()