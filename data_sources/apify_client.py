"""
Apify client for Instagram data scraping.
"""
import os
import time
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any
from apify_client import ApifyClient

from config import APIFY_API_KEY, MAX_REELS_PER_CREATOR, APIFY_CACHE_DIR

class InstagramScraper:
    """Instagram data scraper using Apify."""
    
    def __init__(self, api_key: str = APIFY_API_KEY):
        if not api_key:
            raise RuntimeError("Missing APIFY_API_KEY")
        self.client = ApifyClient(api_key)
        self.cache_dir = Path(APIFY_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def flatten_comments(self, comment_list: List[Dict], max_n: int = 50) -> List[str]:
        """
        Convert a single comment list (Apify objects) → simple text list.
        """
        if not isinstance(comment_list, list):
            return []
        out = []
        for c in comment_list[:max_n]:
            if isinstance(c, dict):
                txt = c.get("text") or c.get("body") or ""
                txt = (txt or "").strip()
                if txt:
                    out.append(txt)
        return out
    
    def load_or_fetch_reels_cached(self, creator: str, max_items: int) -> pd.DataFrame:
        """
        Cache wrapper around fetch_reels_from_apify.
        """
        cache_path = self.cache_dir / f"{creator}_max{max_items}.parquet"

        if cache_path.exists():
            print(f"📂 Using cached Apify reels for @{creator} from {cache_path}")
            return pd.read_parquet(cache_path)

        # Fallback: real network call once
        df = self.fetch_reels_from_apify(creator, max_items=max_items)
        if not df.empty:
            df.to_parquet(cache_path, index=False)
            print(f"💾 Cached Apify reels for @{creator} → {cache_path}")
        else:
            print(f"⚠️ No reels for @{creator}, nothing cached.")
        return df
    
    def fetch_reels_from_apify(self, handle: str, max_items: int = MAX_REELS_PER_CREATOR) -> pd.DataFrame:
        """
        NO-CACHE reels fetch from Apify.
        """
        print(f"\n📸 Fetching reels for @{handle} via Apify...")

        @contextmanager
        def _t(name: str, **meta):
            t0 = time.perf_counter()
            try:
                yield
            finally:
                dt = time.perf_counter() - t0
                # Optional: record timing if needed
                pass

        try:
            run_input = {
                "username": [handle],
                "resultsLimit": int(max_items),
            }

            with _t("apify.actor.call", actor_id="xMc5Ga1oCONPmWJIa"):
                run = self.client.actor("xMc5Ga1oCONPmWJIa").call(run_input=run_input)

            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                print("  ✗ Missing defaultDatasetId from Apify run.")
                return pd.DataFrame()

            with _t("apify.dataset.list_items", dataset_id=dataset_id):
                items = self.client.dataset(dataset_id).list_items().items

            if not items:
                print("  ✗ No items returned.")
                return pd.DataFrame()

            with _t("items_to_df"):
                df = pd.DataFrame(items)

            with _t("normalize_fields"):
                # reel_url
                if "url" in df.columns:
                    df["reel_url"] = df["url"]
                elif "shortcode" in df.columns:
                    df["reel_url"] = "https://www.instagram.com/reel/" + df["shortcode"].astype(str) + "/"
                else:
                    df["reel_url"] = None

                # caption
                if "caption" in df.columns:
                    df["caption_norm"] = df["caption"].fillna("")
                else:
                    df["caption_norm"] = ""

            # comments (including deep)
            comment_fields = [
                "latestComments",
                "comments",
                "deepLatestComments",
                "deepComments",
            ]
            comment_fields = [c for c in comment_fields if c in df.columns]

            with _t("flatten_comments", fields=",".join(comment_fields) if comment_fields else ""):
                if comment_fields:
                    def collect_all_comments(row, max_total=100):
                        texts = []
                        for col in comment_fields:
                            texts.extend(self.flatten_comments(row.get(col), max_n=50))
                        # de-duplicate while preserving order
                        seen, uniq = set(), []
                        for t in texts:
                            if t and t not in seen:
                                seen.add(t)
                                uniq.append(t)
                            if len(uniq) >= max_total:
                                break
                        return uniq

                    df["flat_comments"] = df.apply(collect_all_comments, axis=1)
                else:
                    df["flat_comments"] = [[] for _ in range(len(df))]

            with _t("url_filter"):
                mask = (
                    df["reel_url"].notna()
                    & (
                        df["reel_url"].astype(str).str.contains("/reel/", na=False)
                        | df["reel_url"].astype(str).str.contains("/p/", na=False)
                    )
                )

            out = df.loc[mask, ["reel_url", "caption_norm", "flat_comments"]].copy()
            out = out.rename(columns={"caption_norm": "caption"}).reset_index(drop=True)

            print(f"  ✓ {len(out)} valid reels for @{handle}")
            return out

        except Exception as e:
            print(f"  ✗ Apify error for @{handle}: {e}")
            return pd.DataFrame()

# Global scraper instance
instagram_scraper = InstagramScraper()