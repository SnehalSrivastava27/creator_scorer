from apify_client import ApifyClient
import pandas as pd
import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import requests

# --- CONFIGURATION ---
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
INPUT_CSV = "train_data.csv"              
OUTPUT_CSV = "background_scores_robust.csv"
TEMP_DIR = "./temp_bg_content"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_ITEMS = 12  # Increased from 5 to 12
FRAMES_PER_VIDEO = 3  # Check Start, Middle, End of video

# --- 1. LOAD SEGMENTATION MODEL ---
print("Loading Segmentation Model...")
seg_model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT').to(DEVICE)
seg_model.eval()

seg_transform = transforms.Compose([
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. ROBUST BACKGROUND ANALYZER ---
def analyze_background(image_path):
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return None
        
        # Resize if huge to save processing time
        if img_bgr.shape[1] > 1000:
            scale = 1000 / img_bgr.shape[1]
            img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # A. GET PERSON MASK
        input_tensor = seg_transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = seg_model(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        mask_resized = cv2.resize(output_predictions, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Class 15 = Person. Create mask where 1=Background, 0=Person
        is_person = (mask_resized == 15)
        
        # LOGIC UPDATE: If < 5% is person, treat WHOLE image as background (Scenery Shot)
        if np.sum(is_person) / is_person.size < 0.05:
            bg_mask = np.ones_like(mask_resized, dtype=np.uint8)
        else:
            bg_mask = (mask_resized != 15).astype(np.uint8)

        # Skip if screen is > 90% person (Extreme Close-up)
        if np.sum(bg_mask) / bg_mask.size < 0.1: 
            return None

        # B. CALCULATE METRICS
        
        # 1. CLUTTER (Canny Edge Density)
        edges = cv2.Canny(img_bgr, 100, 200)
        bg_edges = cv2.bitwise_and(edges, edges, mask=bg_mask)
        # Normalize: Edges per pixel
        clutter_score = np.sum(bg_edges) / np.sum(bg_mask)
        
        # 2. BRIGHTNESS & SATURATION (HSV)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        sat_score = cv2.mean(s, mask=bg_mask)[0]
        bright_score = cv2.mean(v, mask=bg_mask)[0]
        
        return {
            "clutter": clutter_score,
            "saturation": sat_score,
            "brightness": bright_score
        }

    except: return None

# --- 3. DOWNLOAD & PROCESS ---
def get_creator_list(csv_path):
    if not os.path.exists(csv_path): return []
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df['creator'].unique().tolist() if 'creator' in df.columns else []

def get_content_map(creators):
    client = ApifyClient(APIFY_TOKEN)
    profile_urls = [f"https://www.instagram.com/{u}/" for u in creators]
    
    print(f"🚀 Scraping {MAX_ITEMS} items for {len(creators)} creators...")
    run_input = {
        "directUrls": profile_urls,
        "resultsType": "posts",
        "resultsLimit": MAX_ITEMS,
        "searchLimit": 1,
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]}
    }
    run = client.actor("apify/instagram-scraper").call(run_input=run_input)
    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
    
    content_map = {c: [] for c in creators}
    for item in dataset_items:
        creator = item.get("ownerUsername")
        if not creator: continue
        if creator not in content_map: content_map[creator] = []
        
        if item.get("videoUrl"):
            content_map[creator].append({"type": "video", "url": item.get("videoUrl")})
        elif item.get("displayUrl"):
            content_map[creator].append({"type": "image", "url": item.get("displayUrl")})
    return content_map

def download_file(url, path):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(1024): f.write(chunk)
            return True
    except: pass
    return False

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    
    creators = get_creator_list(INPUT_CSV)
    content_map = get_content_map(creators)
    
    final_results = []
    print("\n--- Starting Robust Background Analysis ---")
    
    for creator, items in content_map.items():
        print(f"--> @{creator}")
        metrics = []
        
        for i, item in enumerate(items):
            ext = "mp4" if item['type'] == "video" else "jpg"
            path = os.path.join(TEMP_DIR, f"{creator}_{i}.{ext}")
            
            if download_file(item['url'], path):
                
                # --- STRATEGY: MULTI-FRAME SAMPLING ---
                frames_to_check = []
                
                if item['type'] == 'video':
                    cap = cv2.VideoCapture(path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 10:
                        # Grab at 20%, 50%, 80% to see different angles
                        points = [0.2, 0.5, 0.8]
                        for p in points:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * p))
                            ret, fr = cap.read()
                            if ret:
                                fname = f"temp_{i}_{p}.jpg"
                                cv2.imwrite(fname, fr)
                                frames_to_check.append(fname)
                    cap.release()
                else:
                    frames_to_check.append(path)
                
                # Analyze all sampled frames
                for fpath in frames_to_check:
                    m = analyze_background(fpath)
                    if m: metrics.append(m)
                    # Cleanup frame
                    if fpath != path and os.path.exists(fpath): os.remove(fpath)
                
                # Cleanup main file
                os.remove(path)

        # Average Results
        if metrics:
            df_m = pd.DataFrame(metrics)
            res = {
                "creator": creator,
                "bg_clutter_score": df_m['clutter'].mean(),
                "bg_brightness_score": df_m['brightness'].mean(),
                "bg_saturation_score": df_m['saturation'].mean(),
                "bg_samples_count": len(metrics)
            }
            final_results.append(res)
            print(f"    >> Avg Clutter: {res['bg_clutter_score']:.2f} | Samples: {len(metrics)}")
        else:
            print("    >> SKIPPED (No valid backgrounds)")
            final_results.append({"creator": creator})

    # Save
    pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Scores saved to {OUTPUT_CSV}")