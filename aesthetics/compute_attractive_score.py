from apify_client import ApifyClient
import pandas as pd
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import requests
from beauty_model import BetaVAE

# --- CONFIGURATION ---
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
INPUT_CSV = "train_data.csv"                  # <--- Your Training Data File
OUTPUT_CSV = "final_aesthetic_scores.csv"
MODEL_PATH = "beta_vae_utkface.pth"
GOLD_VECTOR_PATH = "gold_standard_female.pth"
TEMP_DIR = "./temp_content"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_ITEMS_PER_CREATOR = 10 
MIN_FACE_SIZE = 90       
MIN_SAMPLES_REQUIRED = 3 

# --- 2. GET CREATORS FROM CSV ---
def get_creator_list(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        exit()
    
    df = pd.read_csv(csv_path)
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    if 'creator' not in df.columns:
        print("❌ Error: CSV must have a 'creator' column.")
        exit()
        
    creators = df['creator'].unique().tolist()
    print(f"✅ Loaded {len(creators)} creators from {csv_path}")
    return creators

# --- 3. DOWNLOADER ---
def get_content_map(creators):
    client = ApifyClient(APIFY_TOKEN)
    profile_urls = [f"https://www.instagram.com/{u}/" for u in creators]
    
    print(f"🚀 Scraping content for {len(creators)} creators...")
    
    run_input = {
        "directUrls": profile_urls,
        "resultsType": "posts",
        "resultsLimit": MAX_ITEMS_PER_CREATOR,
        "searchLimit": 1,
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]}
    }
    
    # Run Apify Actor
    run = client.actor("apify/instagram-scraper").call(run_input=run_input)
    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
    
    content_map = {c: [] for c in creators}
    
    for item in dataset_items:
        creator = item.get("ownerUsername")
        if not creator: continue
        
        # Handle new creators found by redirect
        if creator not in content_map:
            content_map[creator] = []
        
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

# --- 4. SCORING ENGINE ---
def process_content(path, content_type, model, gold_vec, transform, face_cascade):
    scores = []
    
    if content_type == "video":
        cap = cv2.VideoCapture(path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            count += 1
            if count % 15 != 0: continue 
            s = score_frame(frame, model, gold_vec, transform, face_cascade)
            if s: scores.append(s)
        cap.release()

    elif content_type == "image":
        frame = cv2.imread(path)
        if frame is not None:
            s = score_frame(frame, model, gold_vec, transform, face_cascade)
            if s: scores.append(s)

    return scores

def score_frame(img_bgr, model, gold_vec, transform, face_cascade):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0: return None
    
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    
    if w < MIN_FACE_SIZE: return None 
    
    roi_gray = gray[y:y+h, x:x+w]
    if cv2.Laplacian(roi_gray, cv2.CV_64F).var() < 80: return None

    m = int(w * 0.3)
    y0, y1 = max(0, y-m), min(img_bgr.shape[0], y+h+m)
    x0, x1 = max(0, x-m), min(img_bgr.shape[1], x+w+m)
    face_img = img_bgr[y0:y1, x0:x1]

    try:
        pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        t = transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): _, mu, _ = model(t)
        dist = torch.norm(mu[0] - gold_vec).item()
        return 10 / (1 + (dist / 15))
    except: return None

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    
    # 1. Load Models
    vae = BetaVAE().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        vae.eval()
    else:
        print("Model missing.")
        exit()
    gold_vec = torch.load(GOLD_VECTOR_PATH, map_location=DEVICE)
    trans = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 2. Get Creators from CSV
    creator_list = get_creator_list(INPUT_CSV)

    # 3. Get Content via Apify
    content_map = get_content_map(creator_list)
    
    results = []
    print("\n--- Scoring Pipeline ---")
    
    for creator, items in content_map.items():
        print(f"--> @{creator}")
        all_scores = []
        
        for i, item in enumerate(items):
            ext = "mp4" if item['type'] == "video" else "jpg"
            path = os.path.join(TEMP_DIR, f"{creator}_{i}.{ext}")
            
            if download_file(item['url'], path):
                item_scores = process_content(path, item['type'], vae, gold_vec, trans, face)
                if item_scores:
                    all_scores.extend(item_scores)
                    print(f"    Item {i}: Found {len(item_scores)} valid frames")
                else:
                    print(f"    Item {i}: Skipped (No valid face)")
                os.remove(path)
        
        sample_count = len(all_scores)
        if sample_count >= MIN_SAMPLES_REQUIRED:
            avg = np.mean(all_scores)
            print(f"    >> FINAL: {avg:.2f} ({sample_count} frames)")
        else:
            avg = None
            print(f"    >> SKIPPED (Only {sample_count} frames)")

        results.append({
            'creator': creator,
            'aesthetic_score': avg,
            'face_sample_count': sample_count
        })

    # 4. Save & Normalize
    df = pd.DataFrame(results)
    
    if not df.empty and df['aesthetic_score'].notna().any():
        valid = df.loc[df['aesthetic_score'].notna()]
        mean, std = valid['aesthetic_score'].mean(), valid['aesthetic_score'].std()
        
        df.loc[valid.index, 'normalized'] = (valid['aesthetic_score'] - mean) / std
        
        min_v = df['normalized'].min()
        max_v = df['normalized'].max()
        
        # Stretch to 1-10
        df.loc[valid.index, 'stretched_score_0_10'] = 1 + (9 * (df.loc[valid.index, 'normalized'] - min_v) / (max_v - min_v))
        
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Scores saved to {OUTPUT_CSV}")