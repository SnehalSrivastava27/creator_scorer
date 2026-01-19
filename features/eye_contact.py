"""
Eye contact detection module.
Analyzes video frames to determine eye contact ratio and score.
"""
import os
import cv2
import numpy as np
from typing import List, Dict

from utils.video_utils import sample_frames_from_video
from config import FRAME_SAMPLE_COUNT

# Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

class EyeContactAnalyzer:
    """Analyzes eye contact in video reels."""
    
    def __init__(self, frames_per_reel: int = FRAME_SAMPLE_COUNT):
        self.frames_per_reel = frames_per_reel
    
    def is_eye_contact_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        True if frame looks like creator is facing camera:
          - frontal-ish face
          - at least 2 reasonably aligned eyes
        """
        if frame_bgr is None:
            return False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return False

        # Check the largest face for eyes
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Extract face region
        face_gray = gray[y:y+h, x:x+w]

        # Detect eyes within the face
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(eyes) < 2:
            return False

        # Check if eyes are reasonably aligned (similar y-coordinates)
        eye_centers = [(ex + ew//2, ey + eh//2) for ex, ey, ew, eh in eyes]
        
        if len(eye_centers) >= 2:
            # Sort by x-coordinate to get left and right eyes
            eye_centers.sort(key=lambda p: p[0])
            left_eye, right_eye = eye_centers[0], eye_centers[-1]
            
            # Check vertical alignment (y-coordinates should be similar)
            y_diff = abs(left_eye[1] - right_eye[1])
            face_height = h
            
            # Eyes should be within 15% of face height vertically
            if y_diff < 0.15 * face_height:
                return True

        return False
    
    def compute_eye_contact_for_reel(self, video_path: str) -> Dict[str, float]:
        """
        Compute eye contact metrics for a video reel.
        
        Returns:
            Dict with eye_contact_ratio and eye_contact_score_0_10
        """
        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return {
                "eye_contact_ratio": np.nan,
                "eye_contact_score_0_10": np.nan,
            }

        # Sample frames from video
        frames = sample_frames_from_video(video_path, max_frames=self.frames_per_reel)
        
        if not frames:
            print("    ✗ No frames sampled from video")
            return {
                "eye_contact_ratio": np.nan,
                "eye_contact_score_0_10": np.nan,
            }

        # Count frames with eye contact
        eye_contact_frames = 0
        total_frames = len(frames)

        for frame in frames:
            if self.is_eye_contact_frame(frame):
                eye_contact_frames += 1

        # Calculate ratio
        eye_contact_ratio = eye_contact_frames / total_frames if total_frames > 0 else 0.0
        
        # Convert to 0-10 score
        eye_contact_score_0_10 = eye_contact_ratio * 10.0

        return {
            "eye_contact_ratio": float(eye_contact_ratio),
            "eye_contact_score_0_10": float(eye_contact_score_0_10),
        }

# Global analyzer instance
eye_contact_analyzer = EyeContactAnalyzer()