"""
Transcript generation module.
Generates transcripts from video audio using Whisper.
"""
import os
import tempfile
from typing import Optional, Dict

from config import DEVICE

# Whisper import
try:
    import whisper
    whisper_model = whisper.load_model("base", device=DEVICE)
    WHISPER_AVAILABLE = True
    print(f"Whisper model loaded on device: {DEVICE}")
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available, transcript generation will be disabled")

class TranscriptAnalyzer:
    """Generates and analyzes transcripts from video audio."""
    
    def __init__(self):
        self.whisper_available = WHISPER_AVAILABLE
    
    def transcribe_reel(self, video_path: str) -> Optional[str]:
        """
        Generate transcript from video audio using Whisper.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Transcript text or None if failed
        """
        if not self.whisper_available:
            print("    ✗ Whisper not available, skipping transcription")
            return None
        
        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return None
        
        try:
            # Transcribe using Whisper
            result = whisper_model.transcribe(video_path)
            transcript = result.get("text", "").strip()
            
            return transcript if transcript else None
            
        except Exception as e:
            print(f"    ✗ Error during transcription: {e}")
            return None
    
    def compute_word_count(self, transcript: str) -> int:
        """
        Count words in transcript, excluding music-related content.
        """
        if not transcript:
            return 0
        
        # Simple word counting (you can make this more sophisticated)
        # Filter out common music indicators
        music_indicators = [
            "[music]", "(music)", "♪", "♫", "🎵", "🎶",
            "[instrumental]", "(instrumental)",
            "[singing]", "(singing)"
        ]
        
        # Remove music indicators
        cleaned_transcript = transcript.lower()
        for indicator in music_indicators:
            cleaned_transcript = cleaned_transcript.replace(indicator, "")
        
        # Count words
        words = cleaned_transcript.split()
        return len(words)
    
    def compute_transcript_metrics_for_reel(self, video_path: str) -> Dict[str, any]:
        """
        Compute transcript-related metrics for a video reel.
        
        Returns:
            Dict with transcript and word count
        """
        transcript = self.transcribe_reel(video_path)
        
        if transcript is None:
            return {
                "transcript": "",
                "word_count": 0,
                "avg_words_spoken_non_music": 0.0,
            }
        
        word_count = self.compute_word_count(transcript)
        
        return {
            "transcript": transcript,
            "word_count": word_count,
            "avg_words_spoken_non_music": float(word_count),  # Per reel average
        }

# Global analyzer instance
transcript_analyzer = TranscriptAnalyzer()