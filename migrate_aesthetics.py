"""
Migration script to move aesthetics files to the proper structure.
Run this script to organize your aesthetics files correctly.
"""
import os
import shutil
from pathlib import Path

def migrate_aesthetics():
    """Migrate aesthetics files to the proper structure."""
    
    print("🔄 Migrating aesthetics files...")
    
    # Create directories if they don't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Files to move to models directory
    model_files = [
        "beta_vae_utkface.pth",
        "gold_standard_female.pth"
    ]
    
    # Check aesthetics directory for model files
    aesthetics_dir = Path("aesthetics")
    if aesthetics_dir.exists():
        print(f"📁 Found aesthetics directory: {aesthetics_dir}")
        
        # Move model files
        for model_file in model_files:
            src_path = aesthetics_dir / model_file
            dst_path = models_dir / model_file
            
            if src_path.exists():
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    print(f"✅ Copied {model_file} to models/")
                else:
                    print(f"⏩ {model_file} already exists in models/")
            else:
                print(f"⚠️ {model_file} not found in aesthetics/")
        
        # List other files in aesthetics directory
        print(f"\n📋 Other files in aesthetics directory:")
        for file_path in aesthetics_dir.iterdir():
            if file_path.is_file() and file_path.name not in model_files:
                print(f"   - {file_path.name}")
    
    else:
        print("❌ aesthetics directory not found")
    
    # Check current directory for model files
    print(f"\n🔍 Checking current directory for model files...")
    for model_file in model_files:
        src_path = Path(model_file)
        dst_path = models_dir / model_file
        
        if src_path.exists():
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                print(f"✅ Moved {model_file} to models/")
            else:
                print(f"⏩ {model_file} already exists in models/")
    
    print(f"\n📊 Final models directory contents:")
    if models_dir.exists():
        for file_path in models_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   - {file_path.name} ({size_mb:.1f} MB)")
    
    print(f"\n✅ Migration complete!")
    print(f"💡 The composite attractiveness analyzer will now use:")
    print(f"   - Enhanced face aesthetic scoring (Beta-VAE)")
    print(f"   - Background quality analysis (DeepLabV3)")
    print(f"   - Composite scoring with proper normalization")

if __name__ == "__main__":
    migrate_aesthetics()