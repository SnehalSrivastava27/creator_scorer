"""
Creativity analysis module.
Measures visual creativity through frame-to-frame changes using multiple metrics.
Also includes face density analysis and outlier detection.
"""
import os
import numpy as np
from typing import Dict, List, Tuple
import cv2
from PIL import Image

from utils.video_utils import sample_uniform_frames_creativity, compute_hist_distance
from config import MAX_FRAMES_PER_REEL, DEVICE

# CLIP imports (assuming available)
try:
    import torch
    import clip
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available, creativity analysis will be limited")

# Load face cascade for face density analysis
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    FACE_CASCADE_AVAILABLE = True
except Exception as e:
    FACE_CASCADE_AVAILABLE = False
    print(f"Warning: Face cascade not available: {e}")

class CreativityAnalyzer:
    """Analyzes visual creativity in video reels using multiple change metrics."""
    
    def __init__(self, max_frames: int = MAX_FRAMES_PER_REEL):
        self.max_frames = max_frames
        self.clip_available = CLIP_AVAILABLE
        self.face_cascade_available = FACE_CASCADE_AVAILABLE
    
    def clip_embed_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Compute CLIP embedding (L2-normalized) for a single frame (BGR).
        Returns 1D numpy vector.
        """
        if not self.clip_available:
            # Return dummy embedding if CLIP not available
            return np.random.randn(512).astype(np.float32)
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        img = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten().astype("float32")
    
    def frame_has_face_haar(self, frame_bgr: np.ndarray, min_face_width_frac: float = 0.06) -> bool:
        """
        Returns True if a face is detected in the frame using Haar cascade.
        
        Args:
            frame_bgr: BGR frame from video
            min_face_width_frac: Minimum face width as fraction of frame width
        """
        if not self.face_cascade_available or frame_bgr is None:
            return False
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        min_face_size = int(w * min_face_width_frac)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_face_size, min_face_size)
        )
        
        return len(faces) > 0
    
    def compute_face_density_for_frames(self, frames: List[np.ndarray], min_face_width_frac: float = 0.06) -> Tuple[int, float]:
        """
        Compute face density across a list of frames.
        
        Returns:
            (n_face_frames, density) where density is the fraction of frames with faces
        """
        if not frames:
            return 0, 0.0
        
        n_face = sum(1 for frame in frames if self.frame_has_face_haar(frame, min_face_width_frac))
        density = n_face / len(frames)
        return n_face, density
    
    def compute_face_density_for_video(self, video_path: str, max_frames: int = 16, min_face_width_frac: float = 0.06) -> Dict[str, float]:
        """
        Compute face density metrics for a video.
        
        Returns:
            Dictionary with face density metrics
        """
        frames = sample_uniform_frames_creativity(video_path, max_frames=max_frames)
        n = len(frames)
        if n == 0:
            return {
                "n_frames_used": 0, 
                "n_face_frames": 0, 
                "face_frame_density": 0.0, 
                "face_density_0_10": 0.0
            }

        n_face, dens = self.compute_face_density_for_frames(frames, min_face_width_frac=min_face_width_frac)
        return {
            "n_frames_used": int(n),
            "n_face_frames": int(n_face),
            "face_frame_density": float(dens),
            "face_density_0_10": round(float(np.clip(dens, 0.0, 1.0) * 10.0), 2),
        }
    
    def compute_three_change_metrics_for_video(self, video_path: str) -> Dict[str, float]:
        """
        For a given video, compute three frame-to-frame change metrics:
          1) Scene-change density (hist-based threshold → approx shot boundaries)
          2) Mean CLIP embedding distance between consecutive frames
          3) Mean histogram distance between consecutive frames
        """
        frames = sample_uniform_frames_creativity(video_path, max_frames=self.max_frames)
        n_frames = len(frames)

        if n_frames < 2:
            return {
                "n_frames_used": n_frames,
                "scene_change_count": 0,
                "scene_change_density": 0.0,
                "scene_score_0_10": 0.0,
                "mean_clip_dist": 0.0,
                "std_clip_dist": 0.0,
                "clip_score_0_10": 0.0,
                "mean_hist_dist": 0.0,
                "std_hist_dist": 0.0,
                "hist_score_0_10": 0.0,
            }

        # METHOD 3: Histogram diffs
        hist_dists = []
        for i in range(1, n_frames):
            d = compute_hist_distance(frames[i - 1], frames[i])
            hist_dists.append(d)
        hist_dists = np.array(hist_dists, dtype=np.float32)

        mean_hist = float(hist_dists.mean())
        std_hist = float(hist_dists.std())

        # Clip to [0,1] before mapping to 0–10
        mean_hist_clipped = float(np.clip(mean_hist, 0.0, 1.0))
        hist_score = round(mean_hist_clipped * 10.0, 2)

        # METHOD 1: Scene-change density
        scene_thresh = 0.5
        scene_changes = int((hist_dists > scene_thresh).sum())
        scene_change_density = scene_changes / float(n_frames - 1)

        # Normalise density (5× scaling heuristic) then map to 0–10
        scene_density_clipped = float(
            np.clip(scene_change_density * 5.0, 0.0, 1.0)
        )
        scene_score = round(scene_density_clipped * 10.0, 2)

        # METHOD 2: CLIP embedding distances
        clip_embs = []
        for f in frames:
            e = self.clip_embed_frame(f)
            clip_embs.append(e)
        clip_embs = np.stack(clip_embs, axis=0)  # [n_frames, d]

        # compute distances between consecutive embeddings
        clip_dists = []
        for i in range(1, n_frames):
            v1 = clip_embs[i - 1]
            v2 = clip_embs[i]
            # Since vectors are normalized, 1 - cosine similarity ∈ [0, 2]
            cos_sim = float(np.dot(v1, v2))
            d = 1.0 - cos_sim
            clip_dists.append(d)
        clip_dists = np.array(clip_dists, dtype=np.float32)

        mean_clip = float(clip_dists.mean())
        std_clip = float(clip_dists.std())

        # Clip-sim distance is usually within [0, 1]; clip to [0,1]
        mean_clip_clipped = float(np.clip(mean_clip, 0.0, 1.0))
        clip_score = round(mean_clip_clipped * 10.0, 2)

        return {
            "n_frames_used": n_frames,
            # scene-based
            "scene_change_count": int(scene_changes),
            "scene_change_density": float(scene_change_density),
            "scene_score_0_10": scene_score,
            # CLIP-based
            "mean_clip_dist": mean_clip,
            "std_clip_dist": std_clip,
            "clip_score_0_10": clip_score,
            # histogram-based
            "mean_hist_dist": mean_hist,
            "std_hist_dist": std_hist,
            "hist_score_0_10": hist_score,
        }
    
    def compute_creativity_for_reel(self, video_path: str) -> Dict[str, float]:
        """
        Joint-pipeline friendly wrapper for creativity analysis.
        Returns the main creativity scores plus face density metrics.
        """
        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return {
                "scene_score_0_10": 0.0,
                "clip_score_0_10": 0.0,
                "hist_score_0_10": 0.0,
                "face_frame_density": 0.0,
                "face_density_0_10": 0.0,
            }

        try:
            # Compute creativity metrics
            metrics = self.compute_three_change_metrics_for_video(video_path)
            
            # Compute face density metrics
            face_metrics = self.compute_face_density_for_video(video_path, max_frames=self.max_frames)
            
        except Exception as e:
            print(f"    ✗ Error during creativity metrics: {repr(e)}")
            metrics = {
                "n_frames_used": 0,
                "scene_change_count": 0,
                "scene_change_density": 0.0,
                "scene_score_0_10": 0.0,
                "mean_clip_dist": 0.0,
                "std_clip_dist": 0.0,
                "clip_score_0_10": 0.0,
                "mean_hist_dist": 0.0,
                "std_hist_dist": 0.0,
                "hist_score_0_10": 0.0,
            }
            face_metrics = {
                "face_frame_density": 0.0,
                "face_density_0_10": 0.0,
            }

        return {
            "scene_score_0_10": metrics["scene_score_0_10"],
            "clip_score_0_10": metrics["clip_score_0_10"],
            "hist_score_0_10": metrics["hist_score_0_10"],
            "face_frame_density": face_metrics["face_frame_density"],
            "face_density_0_10": face_metrics["face_density_0_10"],
        }

# Function-level interface for backward compatibility
def compute_creativity_for_reel(video_path: str) -> Dict[str, float]:
    """
    Function-level interface matching the original notebook structure.
    """
    return creativity_analyzer.compute_creativity_for_reel(video_path)

# Global analyzer instance
creativity_analyzer = CreativityAnalyzer()

def compute_outlier_2sigma_ratio(word_counts: List[float]) -> float:
    """
    Compute the ratio of word counts that are outliers (beyond ±2σ from global mean).
    
    Args:
        word_counts: List of word counts for all reels across all creators
        
    Returns:
        Dictionary with outlier ratios per creator
    """
    if not word_counts:
        return 0.0
    
    # Convert to numpy array and remove NaN values
    wc = np.array(word_counts)
    wc = wc[~np.isnan(wc)]
    
    if len(wc) == 0:
        return 0.0
    
    # Compute global statistics
    mu = float(wc.mean())
    sigma = float(wc.std(ddof=0))
    
    # Define 2σ thresholds
    low_thr = mu - 2.0 * sigma
    high_thr = mu + 2.0 * sigma
    
    # Count outliers
    outliers = np.sum((wc < low_thr) | (wc > high_thr))
    
    return float(outliers / len(wc)) if len(wc) > 0 else 0.0

def compute_creator_outlier_ratios(df_reels) -> Dict[str, float]:
    """
    Compute outlier_2sigma_ratio for each creator based on their word counts.
    
    Args:
        df_reels: DataFrame with all reel data including 'creator' and 'word_count' columns
        
    Returns:
        Dictionary mapping creator names to their outlier ratios
    """
    if df_reels.empty or 'word_count' not in df_reels.columns:
        return {}
    
    # Get all word counts for global statistics
    all_word_counts = df_reels['word_count'].dropna().values
    
    if len(all_word_counts) == 0:
        return {}
    
    # Compute global statistics
    mu = float(all_word_counts.mean())
    sigma = float(all_word_counts.std(ddof=0))
    
    # Define 2σ thresholds
    low_thr = mu - 2.0 * sigma
    high_thr = mu + 2.0 * sigma
    
    print(f"Global word count statistics: μ={mu:.3f}, σ={sigma:.3f}")
    print(f"2σ thresholds: low<{low_thr:.3f}, high>{high_thr:.3f}")
    
    # Compute per-creator outlier ratios
    creator_ratios = {}
    
    for creator in df_reels['creator'].unique():
        creator_data = df_reels[df_reels['creator'] == creator]
        creator_word_counts = creator_data['word_count'].dropna().values
        
        if len(creator_word_counts) == 0:
            creator_ratios[creator] = 0.0
            continue
        
        # Count outliers for this creator
        outliers = np.sum((creator_word_counts < low_thr) | (creator_word_counts > high_thr))
        ratio = float(outliers / len(creator_word_counts))
        
        creator_ratios[creator] = ratio
    
    return creator_ratios