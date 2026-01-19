import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
FACE_SCORES_CSV = "final_aesthetic_scores.csv"
BG_SCORES_CSV = "background_scores_robust.csv"
TRAIN_DATA_CSV = "train_data.csv"

# The Metrics we want to test
METRIC_COLS = [
    "stretched_score_0_10",      # The Face Score (High = Ideal Structure)
    "bg_clutter_score",        # Background Clutter (Low = Clean)
    "bg_brightness_score",     # Background Brightness (High = Airy)
    "bg_saturation_score",     # Background Vibrance (High = Colorful)
    "composite_score"          # We will calculate this below
]

# The Attributes we want to predict (ToFu/BoFu)
TARGET_ATTRS = [
    "aspirational", 
    "cool", 
    "relatable", 
    "credible",
    "communication",
    "story_telling"
]

def load_and_merge():
    # 1. Load Dataframes
    if not os.path.exists(FACE_SCORES_CSV) or not os.path.exists(BG_SCORES_CSV):
        print("❌ Error: Missing score files. Run the scoring scripts first.")
        return None

    df_face = pd.read_csv(FACE_SCORES_CSV)
    df_bg = pd.read_csv(BG_SCORES_CSV)
    df_train = pd.read_csv(TRAIN_DATA_CSV)
    
    # Clean column names
    df_train.columns = [c.strip() for c in df_train.columns]
    
    # 2. Merge Scores
    # Merge Face + Background on 'creator'
    df_scores = pd.merge(df_face, df_bg, on="creator", how="inner")
    
    # 3. Calculate "Composite Score"
    # Logic: High Face Score + High Brightness + Low Clutter = "Perfect Aesthetic"
    # We need to normalize them first to combine them math-appropriately
    
    # Normalize Clutter (Invert it: Lower is better)
    df_scores['norm_clutter'] = (df_scores['bg_clutter_score'] - df_scores['bg_clutter_score'].mean()) / df_scores['bg_clutter_score'].std()
    df_scores['inv_clutter'] = df_scores['norm_clutter'] * -1 
    
    # Normalize Brightness
    df_scores['norm_bright'] = (df_scores['bg_brightness_score'] - df_scores['bg_brightness_score'].mean()) / df_scores['bg_brightness_score'].std()
    
    # Normalize Face Score (using the normalized column if it exists, or calc new)
    if 'normalized' in df_scores.columns:
        face_norm = df_scores['normalized']
    else:
        face_norm = (df_scores['stretched_score_v2'] - df_scores['stretched_score_v2'].mean()) / df_scores['stretched_score_v2'].std()

    # Create Composite (Weighted Average)
    # 50% Face, 25% Brightness, 25% Cleanliness
    df_scores['composite_score'] = (0.5 * face_norm) + (0.25 * df_scores['norm_bright']) + (0.25 * df_scores['inv_clutter'])

    # 4. Merge with Training Data (Targets)
    # Aggregate training data if multiple rows per creator
    df_train_agg = df_train.groupby('creator')[TARGET_ATTRS].mean(numeric_only=True).reset_index()
    
    df_final = pd.merge(df_scores, df_train_agg, on="creator", how="inner")
    
    print(f"✅ Successfully matched {len(df_final)} creators.")
    return df_final

def run_correlation():
    df = load_and_merge()
    if df is None: return

    # Calculate Correlations
    # We want rows = Metrics, cols = Targets
    matrix = df[METRIC_COLS + TARGET_ATTRS].corr(method="pearson")
    
    # Extract just the relevant slice (Metrics vs Targets)
    heatmap_data = matrix.loc[METRIC_COLS, TARGET_ATTRS]
    
    print("\n--- Correlation Results ---")
    print(heatmap_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="RdBu", center=0, vmin=-0.6, vmax=0.6)
    plt.title("Triangulated Aesthetic Score Correlations")
    plt.tight_layout()
    plt.savefig("triangulated_heatmap.png")
    print("\n🎉 Saved heatmap to 'triangulated_heatmap.png'")

if __name__ == "__main__":
    run_correlation()