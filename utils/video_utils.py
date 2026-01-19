"""
Video processing utilities.
"""
import os
import cv2
import numpy as np
from typing import List, Tuple
import math

def sample_frames_from_video(video_path: str, max_frames: int = 16) -> List[np.ndarray]:
    """
    Uniformly sample up to `max_frames` frames from a video.
    Returns a list of frames in BGR (OpenCV default).
    """
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames == 0:
        cap.release()
        return []

    # Choose indices uniformly across the video
    indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
    indices_set = set(indices.tolist())

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices_set:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames

def sample_uniform_frames_creativity(video_path: str, max_frames: int = 16) -> List[np.ndarray]:
    """
    Sample frames uniformly for creativity analysis.
    Alias for sample_frames_from_video for consistency with notebook.
    """
    return sample_frames_from_video(video_path, max_frames)

def sample_bottom_frames(
    video_path: str,
    target_fps: float = 3.0,
    bottom_ratio: float = 0.4,
) -> Tuple[List[Tuple[float, np.ndarray]], float]:
    """
    Open a video, sample frames at target_fps, and return:
      frames:   list of (time_sec, cropped_bottom_frame)
      duration: total video duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if original_fps <= 0.0:
        # fallback: assume 30 fps to avoid div-by-zero
        original_fps = 30.0

    duration = frame_count / original_fps if frame_count > 0 else 0.0

    # how many frames to skip between samples
    frame_step = max(1, int(round(original_fps / max(target_fps, 0.1))))

    frames: List[Tuple[float, np.ndarray]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            h, w = frame.shape[:2]
            y0 = int(h * (1.0 - bottom_ratio))
            cropped = frame[y0:h, 0:w]

            t_sec = frame_idx / original_fps
            frames.append((t_sec, cropped))

        frame_idx += 1

    cap.release()
    return frames, float(duration)

def compute_hist_distance(f1: np.ndarray, f2: np.ndarray, bins: int = 32) -> float:
    """
    Compute Bhattacharyya distance between HSV histograms of two frames.
    """
    f1_hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
    f2_hsv = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)

    h1 = cv2.calcHist(
        [f1_hsv], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    h2 = cv2.calcHist(
        [f2_hsv], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )

    h1 = h1.flatten().astype("float32")
    h2 = h2.flatten().astype("float32")

    h1 /= (h1.sum() + 1e-8)
    h2 /= (h2.sum() + 1e-8)

    dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return float(dist)

def face_cues(frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Compute:
      - face_area_frac: face bounding box area / frame area
      - center_offset_norm: distance between face center and frame center,
        normalized to [0, 1] (0 = perfectly centered, 1 = at extreme corner).
    """
    h, w = frame_shape[:2]
    x, y, bw, bh = bbox

    face_area = float(bw * bh)
    frame_area = float(w * h) if (w > 0 and h > 0) else 1.0
    face_area_frac = face_area / frame_area

    frame_cx, frame_cy = w / 2.0, h / 2.0
    face_cx, face_cy = x + bw / 2.0, y + bh / 2.0

    dx = face_cx - frame_cx
    dy = face_cy - frame_cy
    dist = math.sqrt(dx * dx + dy * dy)

    # Max possible distance is corner to center
    max_dist = math.sqrt(frame_cx ** 2 + frame_cy ** 2) or 1.0
    center_offset_norm = dist / max_dist  # 0 center → 1 corner

    return float(face_area_frac), float(center_offset_norm)