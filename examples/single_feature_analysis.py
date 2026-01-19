"""
Example showing how to use individual feature analyzers.
"""
import os
from features.attractiveness import attractiveness_analyzer
from features.eye_contact import eye_contact_analyzer
from features.creativity import creativity_analyzer
from features.transcript import transcript_analyzer

def analyze_single_video(video_path: str):
    """
    Example of analyzing a single video with individual feature analyzers.
    
    Args:
        video_path: Path to the video file to analyze
    """
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"🎥 Analyzing video: {video_path}")
    print("=" * 50)
    
    # Attractiveness analysis
    print("👁️ Running attractiveness analysis...")
    attractiveness_results = attractiveness_analyzer.compute_attractiveness_for_reel(video_path)
    print(f"   Attractiveness Score: {attractiveness_results.get('multi_cue_attr_0_10', 'N/A'):.2f}/10")
    print(f"   Face Aesthetic: {attractiveness_results.get('aesthetic_face_0_10', 'N/A'):.2f}/10")
    print(f"   Lighting Score: {attractiveness_results.get('lighting', 'N/A'):.2f}")
    
    # Eye contact analysis
    print("\n👀 Running eye contact analysis...")
    eye_contact_results = eye_contact_analyzer.compute_eye_contact_for_reel(video_path)
    print(f"   Eye Contact Score: {eye_contact_results.get('eye_contact_score_0_10', 'N/A'):.2f}/10")
    print(f"   Eye Contact Ratio: {eye_contact_results.get('eye_contact_ratio', 'N/A'):.2f}")
    
    # Creativity analysis
    print("\n🎨 Running creativity analysis...")
    creativity_results = creativity_analyzer.compute_creativity_for_reel(video_path)
    print(f"   Creativity Score: {creativity_results.get('hist_score_0_10', 'N/A'):.2f}/10")
    print(f"   Scene Changes: {creativity_results.get('scene_score_0_10', 'N/A'):.2f}/10")
    print(f"   CLIP Similarity: {creativity_results.get('clip_score_0_10', 'N/A'):.2f}/10")
    
    # Transcript analysis
    print("\n🎤 Running transcript analysis...")
    transcript_results = transcript_analyzer.compute_transcript_metrics_for_reel(video_path)
    transcript_text = transcript_results.get('transcript', '')
    word_count = transcript_results.get('word_count', 0)
    
    print(f"   Word Count: {word_count}")
    if transcript_text:
        print(f"   Transcript Preview: {transcript_text[:100]}...")
    else:
        print("   No transcript generated")
    
    print("\n✅ Analysis completed!")

def main():
    """Main function for single video analysis example."""
    
    # Example video path - replace with your actual video file
    video_path = "path/to/your/video.mp4"
    
    # You can also use a sample video for testing
    # video_path = "sample_videos/test_reel.mp4"
    
    analyze_single_video(video_path)

if __name__ == "__main__":
    main()