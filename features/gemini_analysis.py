"""
Gemini AI analysis module.
Uses Google's Gemini AI to analyze video content and extract semantic features.
"""
import os
import json
from typing import Dict, Any, Optional, List

from config import GEMINI_API_KEY, GEMINI_MODEL

# Gemini import
try:
    from google import genai
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("Gemini client initialized")
    else:
        GEMINI_AVAILABLE = False
        print("Warning: Gemini API key not found")
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini library not available")

class GeminiAnalyzer:
    """Analyzes video content using Google's Gemini AI."""
    
    def __init__(self):
        self.gemini_available = GEMINI_AVAILABLE
        self.model_name = GEMINI_MODEL
    
    def create_analysis_prompt(self, caption: str, transcript: str, comments: List[str]) -> str:
        """
        Create a structured prompt for Gemini analysis.
        """
        comments_text = "\n".join(comments[:20]) if comments else "No comments available"
        
        prompt = f"""
        Analyze this Instagram reel content and provide a JSON response with the following metrics:

        CONTENT:
        Caption: {caption}
        Transcript: {transcript}
        Comments: {comments_text}

        Please analyze and return a JSON object with these exact fields:
        {{
            "is_marketing": 0 or 1 (1 if content is promotional/marketing),
            "is_educational": 0 or 1 (1 if content is educational/informative),
            "is_vlog": 0 or 1 (1 if content is personal vlog/lifestyle),
            "has_humour": 0 or 1 (1 if content contains humor/comedy),
            "genz_word_count": integer (count of Gen-Z slang words like "slay", "periodt", "no cap", etc.),
            "comment_sentiment_counts": {{
                "questioning": integer (comments asking questions),
                "agreeing": integer (comments showing agreement/support),
                "appreciating": integer (comments showing appreciation/compliments),
                "negative": integer (comments with negative sentiment),
                "neutral": integer (neutral comments)
            }}
        }}

        Return only the JSON object, no additional text.
        """
        return prompt
    
    def call_gemini_for_reel(
        self, 
        caption: str, 
        transcript: str, 
        comments: List[str]
    ) -> Optional[str]:
        """
        Call Gemini API to analyze reel content.
        Returns JSON string for backward compatibility with original interface.
        
        Args:
            caption: Reel caption text
            transcript: Video transcript
            comments: List of comment texts
            
        Returns:
            JSON string with analysis results or None if failed
        """
        if not self.gemini_available:
            print("    ✗ Gemini not available, skipping AI analysis")
            return None
        
        try:
            prompt = self.create_analysis_prompt(caption, transcript, comments)
            
            # Call Gemini API
            response = gemini_client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}]}]
            )
            
            # Extract response text
            if response and response.candidates:
                response_text = response.candidates[0].content.parts[0].text.strip()
                
                # Try to parse JSON response
                try:
                    # Clean response text (remove markdown formatting if present)
                    if response_text.startswith("```json"):
                        response_text = response_text.replace("```json", "").replace("```", "").strip()
                    elif response_text.startswith("```"):
                        response_text = response_text.replace("```", "").strip()
                    
                    # Validate JSON by parsing it
                    analysis_result = json.loads(response_text)
                    return response_text  # Return the JSON string
                    
                except json.JSONDecodeError as e:
                    print(f"    ✗ Failed to parse Gemini JSON response: {e}")
                    print(f"    Response text: {response_text[:200]}...")
                    return None
            else:
                print("    ✗ No valid response from Gemini")
                return None
                
        except Exception as e:
            print(f"    ✗ Error calling Gemini API: {e}")
            return None
    
    def compute_gemini_metrics_for_reel(
        self, 
        caption: str, 
        transcript: str, 
        comments: List[str]
    ) -> Dict[str, Any]:
        """
        Compute Gemini-based metrics for a reel.
        
        Args:
            caption: Reel caption text
            transcript: Video transcript  
            comments: List of comment texts
            
        Returns:
            Dict with Gemini analysis metrics
        """
        # Default values
        default_result = {
            "gemini_raw": "{}",
            "gemini_is_marketing": 0,
            "gemini_is_educational": 0,
            "gemini_is_vlog": 0,
            "gemini_has_humour": 0,
            "gemini_genz_word_count": 0,
            "gemini_comment_sentiment_counts.questioning": 0,
            "gemini_comment_sentiment_counts.agreeing": 0,
            "gemini_comment_sentiment_counts.appreciating": 0,
            "gemini_comment_sentiment_counts.negative": 0,
            "gemini_comment_sentiment_counts.neutral": 0,
        }
        
        # Call Gemini API
        analysis = self.call_gemini_for_reel(caption, transcript, comments)
        
        if analysis is None:
            return default_result
        
        # Extract metrics from analysis
        try:
            result = {
                "gemini_raw": json.dumps(analysis),
                "gemini_is_marketing": int(analysis.get("is_marketing", 0)),
                "gemini_is_educational": int(analysis.get("is_educational", 0)),
                "gemini_is_vlog": int(analysis.get("is_vlog", 0)),
                "gemini_has_humour": int(analysis.get("has_humour", 0)),
                "gemini_genz_word_count": int(analysis.get("genz_word_count", 0)),
            }
            
            # Extract comment sentiment counts
            sentiment_counts = analysis.get("comment_sentiment_counts", {})
            result.update({
                "gemini_comment_sentiment_counts.questioning": int(sentiment_counts.get("questioning", 0)),
                "gemini_comment_sentiment_counts.agreeing": int(sentiment_counts.get("agreeing", 0)),
                "gemini_comment_sentiment_counts.appreciating": int(sentiment_counts.get("appreciating", 0)),
                "gemini_comment_sentiment_counts.negative": int(sentiment_counts.get("negative", 0)),
                "gemini_comment_sentiment_counts.neutral": int(sentiment_counts.get("neutral", 0)),
            })
            
            return result
            
        except Exception as e:
            print(f"    ✗ Error processing Gemini analysis: {e}")
            return default_result

# Global analyzer instance
gemini_analyzer = GeminiAnalyzer()

# Function-level interface for backward compatibility
def call_gemini_for_reel(caption: str, transcript: str, comments: List[str]) -> Optional[str]:
    """
    Function-level interface matching the original notebook structure.
    Returns JSON string with analysis results.
    """
    return gemini_analyzer.call_gemini_for_reel(caption, transcript, comments)