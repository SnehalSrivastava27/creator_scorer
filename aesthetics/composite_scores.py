import pandas as pd
import os

# --- CONFIGURATION ---
FACE_FILE = "final_aesthetic_scores.csv"
BG_FILE = "background_scores_robust.csv"
OUTPUT_FILE = "final_combined_aesthetic.csv"

def main():
    # 1. Load the existing files
    if not os.path.exists(FACE_FILE) or not os.path.exists(BG_FILE):
        print("❌ Error: Input CSV files not found.")
        return

    print("Loading data...")
    df_face = pd.read_csv(FACE_FILE)
    df_bg = pd.read_csv(BG_FILE)

    # Clean creator names to ensure matching works
    df_face['creator'] = df_face['creator'].astype(str).str.strip().str.lower()
    df_bg['creator'] = df_bg['creator'].astype(str).str.strip().str.lower()

    # 2. Merge them (Inner join keeps only creators with BOTH scores)
    print("Merging data...")
    df = pd.merge(df_face, df_bg, on="creator", how="inner")
    
    print(f"Matched {len(df)} creators.")

    # 3. Handle Missing Data (Impute with mean if any NaNs exist)
    cols_to_fix = ['stretched_score_0_10', 'bg_brightness_score', 'bg_clutter_score', 'bg_saturation_score']
    for col in cols_to_fix:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # 4. Normalization Logic (Z-Score)
    # We transform values to a common scale (mean=0, std=1) to combine them mathematically
    def get_z_score(series):
        return (series - series.mean()) / (series.std() + 1e-6)

    norm_face = get_z_score(df['stretched_score_0_10'])
    norm_bright = get_z_score(df['bg_brightness_score'])
    
    # Clutter is "bad", so we invert it (multiply Z-score by -1)
    # Low Clutter -> Negative Z -> Becomes Positive (Good)
    norm_clutter_inv = get_z_score(df['bg_clutter_score']) * -1

    # 5. Calculate Composite Score
    # Weighting: 50% Face, 25% Brightness, 25% Cleanliness (Low Clutter)
    df['composite_score'] = (0.5 * norm_face) + (0.25 * norm_bright) + (0.25 * norm_clutter_inv)

    # 6. Select and Rename Columns for Training
    # We keep 'stretched_score_0_10' mapped to 'method6...' just in case legacy code needs it
    final_df = df[[
        'creator', 
        'composite_score', 
        'bg_saturation_score',
        'stretched_score_0_10' 
    ]].copy()

    final_df = final_df.rename(columns={
        'stretched_score_0_10': 'method6_full_aesthetic_0_10'
    })

    # 7. Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Saved to {OUTPUT_FILE}")
    print(final_df.head())

if __name__ == "__main__":
    main()