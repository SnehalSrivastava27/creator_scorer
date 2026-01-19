# Instagram Reel Feature Extraction System

A comprehensive system for extracting and analyzing features from Instagram reels, including visual, audio, and textual analysis using state-of-the-art AI models.

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create missing modules (if needed)
python create_missing_modules.py

# Set up environment variables
cp .env.example .env
# Edit .env with your actual API keys
```

### 2. Prepare Input
Create `new_creators.csv` with creator usernames:
```csv
creator
username1
@username2
https://www.instagram.com/username3/
username4
```

### 3. Run Analysis
```bash
python main.py
```

The system will:
- ✅ **Resume automatically** if interrupted
- ✅ **Cache downloads** to avoid re-downloading
- ✅ **Cache API responses** to save costs
- ✅ **Append results** to `final_creator_scores.csv` as it processes

## Input/Output Structure

### Input Format
- **File**: `new_creators.csv`
- **Required Column**: `creator`
- **Accepts**: usernames, @usernames, or full Instagram URLs

### Output Format
- **File**: `final_creator_scores.csv`
- **Structure**: One row per creator with aggregated metrics

### Key Output Metrics
```csv
creator,eye_contact_avg_score_0_10,series_reel_mean,avg_captioned_reels,avg_word_count,
avg_genz_word_count,marketing_ratio,educational_ratio,vlog_ratio,has_humour_any,
comments_questioning,comments_agreeing,comments_appreciating,comments_negative,
comments_neutral,avg_english_pct,mean_hist_score,mean_clip_score,mean_scene_score
```

## Architecture

### Processing Pipeline
1. **CSV Input** → Read creators from `new_creators.csv`
2. **Resume Check** → Skip already processed creators
3. **Metadata Fetch** → Get reel data from Apify (cached)
4. **Download** → Get video files with multiple fallback methods
5. **Analysis** → Extract all features using async pipeline
6. **Output** → Append results to CSV immediately

### Async Processing
- **Producer Thread**: Downloads reels in parallel
- **Consumer Thread**: Processes videos and extracts features
- **Queue-based**: Efficient memory usage with bounded queues

### Caching Strategy
- **Metadata Cache**: `apify_metadata_cache/` - API responses
- **Video Cache**: `reels_video_cache/` - Downloaded videos
- **Resume Logic**: Automatically skips processed creators

## Features Extracted

### Visual Analysis
- **Eye Contact**: Haar cascade-based face/eye detection → 0-10 score
- **Captions**: OCR-based dynamic caption detection → 0/1 flag
- **Creativity**: CLIP + histogram analysis → multiple scores

### Audio Analysis
- **Transcription**: Whisper medium model → full transcript
- **Word Count**: Spoken words (excluding music) → count
- **Language**: English detection → percentage

### Content Analysis
- **Series Detection**: Regex patterns for episodes/parts → 0/1 flag
- **Gemini AI**: Semantic analysis → multiple classifications
- **Comment Sentiment**: AI-powered sentiment breakdown

### Aggregated Metrics (Per Creator)
- Average scores across all reels
- Ratios and percentages
- Sum totals for comment sentiments
- Music vs non-music content separation

## Configuration

### API Keys Required
```bash
GEMINI_API_KEY=your_gemini_api_key
APIFY_API_KEY=your_apify_api_key
```

### Key Settings (config.py)
```python
# Processing limits
MAX_REELS_PER_CREATOR = 10     # Reels to analyze per creator
DEVICE = "cuda"/"cpu"          # Auto-detected

# Directories
REEL_DOWNLOAD_DIR = "reels_video_cache"
APIFY_CACHE_DIR = "apify_metadata_cache"
```

## Advanced Usage

### Resume Processing
If interrupted, simply run again:
```bash
python main.py  # Automatically resumes from where it left off
```

### Clear Caches
```bash
# Clear video cache
rm -rf reels_video_cache/

# Clear metadata cache  
rm -rf apify_metadata_cache/

# Clear results (start fresh)
rm final_creator_scores.csv
```

### Monitor Progress
```bash
# Check current results
head -n 5 final_creator_scores.csv

# Count processed creators
wc -l final_creator_scores.csv
```

## Project Structure

```
├── config.py                 # Configuration settings
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── utils/                    # Utility modules
│   ├── cache_manager.py      # Caching utilities
│   └── video_utils.py        # Video processing utilities
├── data_sources/             # Data collection modules
│   └── apify_client.py       # Instagram scraping
├── features/                 # Feature extraction modules
│   ├── attractiveness.py     # Visual attractiveness analysis
│   ├── eye_contact.py        # Eye contact detection
│   ├── creativity.py         # Visual creativity metrics
│   ├── sequential.py         # Series/episode detection
│   ├── video_captions.py     # Dynamic caption detection
│   ├── accessories.py        # Object/accessory detection
│   ├── sun_exposure.py       # Lighting analysis
│   ├── transcript.py         # Audio transcription
│   ├── english_detection.py  # Language detection
│   └── gemini_analysis.py    # AI-powered analysis
└── pipeline/                 # Main processing pipeline
    └── feature_extractor.py  # Orchestrates all analyses
```

## Output Features

The system extracts 50+ features per reel, including:

### Visual Features
- `multi_cue_attr_0_10`: Overall attractiveness score (0-10)
- `eye_contact_score_0_10`: Eye contact score (0-10)
- `hist_score_0_10`: Visual creativity score (0-10)
- `sun_exposure_0_10_A`: Sun exposure score (0-10)
- `has_dynamic_captions`: Dynamic caption presence (0/1)
- `avg_clothing_per_reel`: Average clothing items detected
- `avg_jewellery_per_reel`: Average jewelry items detected
- `avg_gadgets_per_reel`: Average gadget items detected

### Text Features
- `transcript`: Full video transcript
- `avg_words_spoken_non_music`: Average spoken words (excluding music)
- `avg_english_pct_non_music`: English content percentage
- `series_flag`: Series/episode content indicator (0/1)
- `episode_number`: Detected episode number (if any)

### AI Analysis Features
- `gemini_is_marketing`: Marketing content indicator (0/1)
- `gemini_is_educational`: Educational content indicator (0/1)
- `gemini_is_vlog`: Vlog content indicator (0/1)
- `gemini_has_humour`: Humor presence indicator (0/1)
- `gemini_genz_word_count`: Gen-Z slang word count
- Comment sentiment breakdowns

## Configuration

Key configuration options in `config.py`:

```python
# Processing limits
MAX_REELS_PER_CREATOR = 1      # Reels to analyze per creator
FRAME_SAMPLE_COUNT = 16        # Frames to sample per reel
MAX_DOWNLOAD_WORKERS = 10      # Parallel download workers

# Analysis thresholds
CAPTION_MIN_COVERAGE = 0.05    # Minimum caption coverage
SIMILARITY_SAME_SEGMENT = 0.6  # Caption segment similarity threshold
```

## Dependencies

### Core Libraries
- `pandas`, `numpy`: Data processing
- `opencv-python`: Computer vision
- `Pillow`: Image processing
- `torch`, `torchvision`: Deep learning

### AI Models
- `transformers`: Hugging Face models
- `clip-by-openai`: CLIP embeddings
- `ultralytics`: YOLO object detection
- `easyocr`: Optical character recognition
- `openai-whisper`: Audio transcription
- `fasttext`: Language identification

### API Clients
- `apify-client`: Instagram scraping
- `google-generativeai`: Gemini AI analysis

## Performance Notes

- **GPU Acceleration**: Automatically uses GPU when available for faster processing
- **Caching**: Intelligent caching of downloaded reels and API responses
- **Parallel Processing**: Multi-threaded processing for multiple creators
- **Memory Management**: Efficient frame sampling and processing

## Limitations

- Requires valid API keys for Apify and Gemini
- Video download functionality needs to be implemented (placeholder provided)
- Some features require specific model files to be downloaded
- Processing time depends on video length and number of reels

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for Whisper and CLIP models
- Google for Gemini AI
- Ultralytics for YOLO
- Meta for FastText
- Apify for Instagram scraping capabilities