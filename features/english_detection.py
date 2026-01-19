"""
English language detection module.
Detects the percentage of English content in transcripts using fastText.
"""
import os
import tempfile
from typing import Dict, List, Tuple
import re

# FastText import
try:
    import fasttext
    
    # Try to load the language identification model
    model_path = "lid.176.ftz"
    if not os.path.exists(model_path):
        print("Downloading fastText language identification model...")
        # You would need to download this model
        # fasttext.util.download_model('en', if_exists='ignore')
        print("Warning: fastText model not found, using fallback detection")
        FASTTEXT_AVAILABLE = False
    else:
        ft_model = fasttext.load_model(model_path)
        FASTTEXT_AVAILABLE = True
        print("FastText language identification model loaded")
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("Warning: FastText not available, English detection will use fallback method")

class EnglishDetectionAnalyzer:
    """Analyzes English language percentage in transcripts."""
    
    def __init__(self):
        self.fasttext_available = FASTTEXT_AVAILABLE
        
        # Common English words for fallback detection
        self.common_english_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
            'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
            'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
            'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people',
            'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only',
            'come', 'its', 'over', 'think', 'also', 'back', 'after',
            'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give',
            'day', 'most', 'us'
        }
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words, removing punctuation and converting to lowercase.
        """
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def detect_english_fasttext(self, text: str) -> float:
        """
        Detect English percentage using fastText language identification.
        """
        if not self.fasttext_available or not text.strip():
            return 0.0
        
        try:
            # Predict language for the entire text
            predictions = ft_model.predict(text.replace('\n', ' '), k=1)
            
            # Check if English is detected
            if predictions[0][0] == '__label__en':
                confidence = float(predictions[1][0])
                return confidence
            else:
                return 0.0
                
        except Exception as e:
            print(f"    ✗ Error in fastText detection: {e}")
            return 0.0
    
    def detect_english_fallback(self, text: str) -> float:
        """
        Fallback English detection using common English words.
        """
        if not text.strip():
            return 0.0
        
        words = self.tokenize_text(text)
        if not words:
            return 0.0
        
        english_words = sum(1 for word in words if word in self.common_english_words)
        english_percentage = english_words / len(words)
        
        return english_percentage
    
    def detect_english_percentage(self, transcript: str) -> float:
        """
        Detect the percentage of English content in a transcript.
        
        Args:
            transcript: The transcript text to analyze
            
        Returns:
            Float between 0.0 and 1.0 representing English percentage
        """
        if not transcript or not transcript.strip():
            return 0.0
        
        # Remove music indicators and clean text
        music_indicators = [
            "[music]", "(music)", "♪", "♫", "🎵", "🎶",
            "[instrumental]", "(instrumental)",
            "[singing]", "(singing)"
        ]
        
        cleaned_transcript = transcript.lower()
        for indicator in music_indicators:
            cleaned_transcript = cleaned_transcript.replace(indicator, "")
        
        cleaned_transcript = cleaned_transcript.strip()
        
        if not cleaned_transcript:
            return 0.0
        
        # Use fastText if available, otherwise fallback
        if self.fasttext_available:
            return self.detect_english_fasttext(cleaned_transcript)
        else:
            return self.detect_english_fallback(cleaned_transcript)
    
    def compute_english_metrics_for_reel(self, transcript: str) -> Dict[str, float]:
        """
        Compute English language metrics for a transcript.
        
        Args:
            transcript: The transcript text to analyze
            
        Returns:
            Dict with English percentage metrics
        """
        english_pct = self.detect_english_percentage(transcript)
        
        return {
            "english_pct_non_music": english_pct,
            "avg_english_pct_non_music": english_pct,  # Per reel average
        }

# Global analyzer instance
english_detection_analyzer = EnglishDetectionAnalyzer()