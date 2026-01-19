"""
Cache management utilities for the feature extraction system.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any

class CacheManager:
    """Manages caching for various components of the system."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_manifest(self, manifest_name: str = "reel_manifest.json") -> Dict[str, Any]:
        """Load manifest from cache."""
        cache_path = self.cache_dir / manifest_name
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def save_manifest(self, manifest: Dict[str, Any], manifest_name: str = "reel_manifest.json"):
        """Save manifest to cache."""
        cache_path = self.cache_dir / manifest_name
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    
    def get_cache_path(self, filename: str) -> Path:
        """Get full cache path for a file."""
        return self.cache_dir / filename
    
    def cache_exists(self, filename: str) -> bool:
        """Check if cache file exists."""
        return (self.cache_dir / filename).exists()

# Global cache manager instance
cache_manager = CacheManager()

# Download cache for reels
_download_cache = {}

def download_reel_cached(reel_url: str, reel_no: int, task_id: str = "joint") -> str | None:
    """
    Wrapper around get_files_gem so we only hit your reel downloader ONCE per URL.
    """
    if reel_url in _download_cache:
        return _download_cache[reel_url]
    
    # Placeholder for actual download logic
    # You would implement the actual download logic here
    # out = get_files_gem(REEL_URL=reel_url, REEL_NO=str(reel_no), task_id=task_id)
    # if not out:
    #     return None
    
    # For now, return None to indicate download not implemented
    return None