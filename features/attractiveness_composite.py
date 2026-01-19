"""
Composite Attractiveness Analysis Module.
Combines face aesthetic scoring with background analysis for comprehensive attractiveness metrics.
"""
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple
import logging

from utils.video_utils import sample_frames_from_video
from config import (
    DEVICE, FRAME_SAMPLE_COUNT, BETA_VAE_MODEL_PATH, GOLD_VECTOR_PATH,
    MIN_FACE_SIZE, MIN_SAMPLES_REQUIRED, ATTRACTIVENESS_LATENT_DIM,
    FACE_WEIGHT, BRIGHTNESS_WEIGHT, CLEANLINESS_WEIGHT
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths from config
BETA_VAE_MODEL_PATH = BETA_VAE_MODEL_PATH
GOLD_VECTOR_PATH = GOLD_VECTOR_PATH

class BetaVAE(nn.Module):
    """Beta-VAE architecture for face aesthetic scoring."""
    
    def __init__(self, latent_dim: int = ATTRACTIVENESS_LATENT_DIM):
        super(BetaVAE, self).__init__()
        
        # Encoder Blocks
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_var = nn.Linear(256*4*4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        encoded = self.flatten(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder_input(z).view(-1, 256, 4, 4)
        reconstruction = self.decoder(decoded)
        return reconstruction, mu, logvar

    def extract_features(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        return f1, f2, f3, f4


class CompositeAttractivenessAnalyzer:
    """Analyzes both face aesthetics and background quality for comprehensive attractiveness scoring."""
    
    def __init__(self):
        self.device = DEVICE
        self.min_face_size = MIN_FACE_SIZE
        self.min_samples_required = MIN_SAMPLES_REQUIRED
        
        # Initialize models
        self._load_models()
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Transforms
        self.face_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        self.seg_transform = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_models(self):
        """Load the Beta-VAE and segmentation models."""
        try:
            # Load Beta-VAE for face scoring
            self.beta_vae = BetaVAE().to(self.device)
            if os.path.exists(BETA_VAE_MODEL_PATH):
                self.beta_vae.load_state_dict(torch.load(BETA_VAE_MODEL_PATH, map_location=self.device))
                self.beta_vae.eval()
                logger.info("✅ Beta-VAE model loaded successfully")
            else:
                logger.warning(f"⚠️ Beta-VAE model not found at {BETA_VAE_MODEL_PATH}")
                self.beta_vae = None
            
            # Load gold standard vector
            if os.path.exists(GOLD_VECTOR_PATH):
                self.gold_vector = torch.load(GOLD_VECTOR_PATH, map_location=self.device)
                logger.info("✅ Gold standard vector loaded successfully")
            else:
                logger.warning(f"⚠️ Gold vector not found at {GOLD_VECTOR_PATH}")
                self.gold_vector = None
            
            # Load segmentation model for background analysis
            self.seg_model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT').to(self.device)
            self.seg_model.eval()
            logger.info("✅ Segmentation model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.beta_vae = None
            self.gold_vector = None
            self.seg_model = None

    def score_face_in_frame(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Score face aesthetics in a single frame using Beta-VAE."""
        if self.beta_vae is None or self.gold_vector is None:
            return None
            
        try:
            # Detect faces
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            
            if w < self.min_face_size:
                return None
            
            # Check sharpness
            roi_gray = gray[y:y+h, x:x+w]
            if cv2.Laplacian(roi_gray, cv2.CV_64F).var() < 80:
                return None
            
            # Extract face with margin
            margin = int(w * 0.3)
            y0, y1 = max(0, y-margin), min(frame_bgr.shape[0], y+h+margin)
            x0, x1 = max(0, x-margin), min(frame_bgr.shape[1], x+w+margin)
            face_img = frame_bgr[y0:y1, x0:x1]
            
            # Convert to PIL and transform
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            tensor = self.face_transform(pil_img).unsqueeze(0).to(self.device)
            
            # Get embedding and compute distance to gold standard
            with torch.no_grad():
                _, mu, _ = self.beta_vae(tensor)
                distance = torch.norm(mu[0] - self.gold_vector).item()
                
            # Convert distance to 0-10 score
            score = 10 / (1 + (distance / 15))
            return float(score)
            
        except Exception as e:
            logger.debug(f"Face scoring error: {e}")
            return None

    def analyze_background(self, frame_bgr: np.ndarray) -> Optional[Dict[str, float]]:
        """Analyze background quality metrics."""
        if self.seg_model is None:
            return None
            
        try:
            # Resize if too large
            if frame_bgr.shape[1] > 1000:
                scale = 1000 / frame_bgr.shape[1]
                frame_bgr = cv2.resize(frame_bgr, None, fx=scale, fy=scale)

            # Convert to RGB for segmentation
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Get person mask using segmentation
            input_tensor = self.seg_transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.seg_model(input_tensor)['out'][0]
            
            output_predictions = output.argmax(0).byte().cpu().numpy()
            mask_resized = cv2.resize(
                output_predictions, 
                (frame_bgr.shape[1], frame_bgr.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Class 15 = Person. Create background mask
            is_person = (mask_resized == 15)
            
            # If < 5% is person, treat whole image as background (scenery shot)
            if np.sum(is_person) / is_person.size < 0.05:
                bg_mask = np.ones_like(mask_resized, dtype=np.uint8)
            else:
                bg_mask = (mask_resized != 15).astype(np.uint8)

            # Skip if > 90% person (extreme close-up)
            if np.sum(bg_mask) / bg_mask.size < 0.1:
                return None

            # Calculate background metrics
            
            # 1. Clutter (edge density)
            edges = cv2.Canny(frame_bgr, 100, 200)
            bg_edges = cv2.bitwise_and(edges, edges, mask=bg_mask)
            clutter_score = np.sum(bg_edges) / np.sum(bg_mask)
            
            # 2. Brightness and saturation
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            saturation_score = cv2.mean(s, mask=bg_mask)[0]
            brightness_score = cv2.mean(v, mask=bg_mask)[0]
            
            return {
                "bg_clutter_score": float(clutter_score),
                "bg_saturation_score": float(saturation_score),
                "bg_brightness_score": float(brightness_score)
            }
            
        except Exception as e:
            logger.debug(f"Background analysis error: {e}")
            return None

    def compute_attractiveness_for_reel(self, video_path: str) -> Dict[str, float]:
        """Compute comprehensive attractiveness metrics for a reel."""
        if not os.path.exists(video_path):
            return self._get_default_scores()
        
        try:
            # Sample frames from video
            frames = sample_frames_from_video(video_path, max_frames=FRAME_SAMPLE_COUNT)
            
            if not frames:
                return self._get_default_scores()
            
            face_scores = []
            bg_metrics = []
            
            # Analyze each frame
            for frame in frames:
                # Face scoring
                face_score = self.score_face_in_frame(frame)
                if face_score is not None:
                    face_scores.append(face_score)
                
                # Background analysis
                bg_result = self.analyze_background(frame)
                if bg_result is not None:
                    bg_metrics.append(bg_result)
            
            # Compute averages
            avg_face_score = np.mean(face_scores) if face_scores else 0.0
            
            if bg_metrics:
                avg_clutter = np.mean([m["bg_clutter_score"] for m in bg_metrics])
                avg_brightness = np.mean([m["bg_brightness_score"] for m in bg_metrics])
                avg_saturation = np.mean([m["bg_saturation_score"] for m in bg_metrics])
            else:
                avg_clutter = avg_brightness = avg_saturation = 0.0
            
            # Compute composite score using Z-score normalization
            composite_score = self._compute_composite_score(
                avg_face_score, avg_brightness, avg_clutter
            )
            
            return {
                "face_aesthetic_score_0_10": float(avg_face_score),
                "bg_clutter_score": float(avg_clutter),
                "bg_brightness_score": float(avg_brightness),
                "bg_saturation_score": float(avg_saturation),
                "composite_aesthetic_score": float(composite_score),
                "face_samples_count": len(face_scores),
                "bg_samples_count": len(bg_metrics),
                "total_frames_analyzed": len(frames)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing attractiveness for {video_path}: {e}")
            return self._get_default_scores()

    def _compute_composite_score(self, face_score: float, brightness: float, clutter: float) -> float:
        """Compute composite aesthetic score with proper normalization."""
        try:
            # Simple composite without Z-score (since we don't have population stats)
            # Normalize clutter (lower is better, so invert)
            normalized_clutter = max(0, 10 - (clutter * 10))  # Rough normalization
            normalized_brightness = min(10, brightness / 25.5)  # HSV brightness is 0-255
            
            # Weighted combination using config weights
            composite = (FACE_WEIGHT * face_score) + (BRIGHTNESS_WEIGHT * normalized_brightness) + (CLEANLINESS_WEIGHT * normalized_clutter)
            
            return max(0.0, min(10.0, composite))
            
        except Exception as e:
            logger.debug(f"Composite score calculation error: {e}")
            return 5.0  # Default middle score

    def _get_default_scores(self) -> Dict[str, float]:
        """Return default scores when analysis fails."""
        return {
            "face_aesthetic_score_0_10": 0.0,
            "bg_clutter_score": 0.0,
            "bg_brightness_score": 0.0,
            "bg_saturation_score": 0.0,
            "composite_aesthetic_score": 0.0,
            "face_samples_count": 0,
            "bg_samples_count": 0,
            "total_frames_analyzed": 0
        }


# Global analyzer instance
composite_attractiveness_analyzer = CompositeAttractivenessAnalyzer()