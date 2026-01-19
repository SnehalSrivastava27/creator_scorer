"""
Creativity analysis module.
Measures visual creativity through frame-to-frame changes using multiple metrics.
"""
import os
import numpy as np
from typing import Dict, List
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

class CreativityAnalyzer:
    """Analyzes visual creativity in video reels using multiple change metrics."""
    
    def __init__(self, max_frames: int = MAX_FRAMES_PER_REEL):
        self.max_frames = max_frames
        self.clip_available = CLIP_AVAILABLE
    
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
        Returns the main creativity scores matching the expected interface.
        """
        if not video_path or not os.path.exists(video_path):
            print("    ✗ Video path does not exist:", video_path)
            return {
                "scene_score_0_10": 0.0,
                "clip_score_0_10": 0.0,
                "hist_score_0_10": 0.0,
            }

        try:
            metrics = self.compute_three_change_metrics_for_video(video_path)
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

        return {
            "scene_score_0_10": metrics["scene_score_0_10"],
            "clip_score_0_10": metrics["clip_score_0_10"],
            "hist_score_0_10": metrics["hist_score_0_10"],
        }

# Function-level interface for backward compatibility
def compute_creativity_for_reel(video_path: str) -> Dict[str, float]:
    """
    Function-level interface matching the original notebook structure.
    """
    return creativity_analyzer.compute_creativity_for_reel(video_path)

# Global analyzer instance
creativity_analyzer = CreativityAnalyzer()