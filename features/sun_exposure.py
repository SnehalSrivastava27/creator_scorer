"""
Sun exposure analysis module.
Analyzes lighting conditions and sun exposure in video frames.
"""
import os
import cv2
import numpy as np
from typing import Dict, List

from utils.video_utils import sample_frames_from_video
from config import FRAME_SAMPLE_COUNT

class SunExposureAnalyzer:
    """Analyzes sun exposure and lighting conditions in video frames."""
    
    def __init__(self, frames_per_reel: int = FRAME_SAMPLE_COUNT):
        self.frames_per_reel = frames_per_reel
    
    def compute_sun_exposure_raw(self, frame_bgr: np.ndarray) -> float:
        """
        Compute raw sun exposure metric for a single frame.
        Based on brightness and color temperature analysis.
        """
        # Convert to HSV for better analysis
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Get brightness (V channel)
        brightness = hsv[:, :, 2].astype(np.float32)
        mean_brightness = float(brightness.mean())
        
        # Get saturation (S channel) 
        saturation = hsv[:, :, 1].astype(np.float32)
        mean_saturation = float(saturation.mean())
        
        # Convert to LAB for better color analysis
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)
        
        # Compute color temperature proxy (yellow-blue balance)
        # Higher b values indicate warmer (more yellow/sunny) conditions
        mean_b = float(b_channel.mean())
        
        # Combine metrics for sun exposure
        # Higher brightness + higher yellow component + moderate saturation = more sun exposure
        brightness_norm = mean_brightness / 255.0  # 0-1
        saturation_norm = mean_saturation / 255.0  # 0-1
        warmth_norm = (mean_b - 128) / 127.0  # -1 to 1, normalized around neutral
        
        # Weighted combination (you can adjust these weights)
        sun_exposure_raw = (
            0.5 * brightness_norm +
            0.3 * max(0, warmth_norm) +  # Only positive warmth contributes
            0.2 * saturation_norm
        ) * 10.0  # Scale to 0-10 range
        
        return float(max(0.0, sun_exposure_raw))
    
    def compute_sun_exposure_for_reel(self, video_path: str) -> Dict[str, float]:
        """
        Compute sun exposure metrics for a video reel.
        
        Returns:
            Dict with sun_exposure_raw_A and sun_exposure_0_10_A
        """
        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return {
                "sun_exposure_raw_A": np.nan,
                "sun_exposure_0_10_A": np.nan,
            }
        
        # Sample frames from video
        frames = sample_frames_from_video(video_path, max_frames=self.frames_per_reel)
        
        if not frames:
            print("    ✗ No frames sampled from video")
            return {
                "sun_exposure_raw_A": np.nan,
                "sun_exposure_0_10_A": np.nan,
            }
        
        # Compute sun exposure for each frame
        sun_exposures = []
        for frame in frames:
            exposure = self.compute_sun_exposure_raw(frame)
            sun_exposures.append(exposure)
        
        # Calculate average
        mean_sun_exposure = float(np.mean(sun_exposures))
        
        # Normalize to 0-10 scale (already in that range, but ensure bounds)
        sun_exposure_0_10 = float(np.clip(mean_sun_exposure, 0.0, 10.0))
        
        return {
            "sun_exposure_raw_A": mean_sun_exposure,
            "sun_exposure_0_10_A": sun_exposure_0_10,
        }

# Global analyzer instance
sun_exposure_analyzer = SunExposureAnalyzer()