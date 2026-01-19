"""
Sequential content detection module.
Detects series/episode patterns in captions and transcripts.
"""
import re
from typing import Optional, Dict

from config import SERIES_KEYWORDS, EP_PATTERNS

class SequentialAnalyzer:
    """Analyzes sequential content patterns in text."""
    
    def __init__(self):
        self.series_keywords = SERIES_KEYWORDS
        self.ep_patterns = EP_PATTERNS
    
    def clean(self, t: Optional[str]) -> str:
        """Lowercase and collapse whitespace."""
        return re.sub(r"\s+", " ", t.lower()).strip() if isinstance(t, str) else ""
    
    def detect_series_from_text(self, caption: str, transcript: str) -> Dict[str, Optional[int]]:
        """
        Detect series patterns in caption and transcript text.
        
        Args:
            caption: Raw reel caption string
            transcript: Whisper transcript string
        
        Returns:
            Dict with series_flag, matched_keywords, and episode_number
        """
        c = self.clean(caption)
        t = self.clean(transcript)
        combined = c + " " + t

        # Keyword hit detection
        matched = [kw for kw in self.series_keywords if re.search(kw, combined)]
        is_series = len(matched) > 0

        # Extract episode/part number
        epi = None
        for p in self.ep_patterns:
            m = re.search(p, combined)
            if m:
                try:
                    epi = int(m.group(1))
                    break
                except:
                    # bare except to mirror notebook behaviour
                    pass

        return {
            "series_flag": 1 if is_series else 0,
            "matched_keywords": "|".join(matched) if matched else "",
            "episode_number": epi,
        }

# Global analyzer instance
sequential_analyzer = SequentialAnalyzer()