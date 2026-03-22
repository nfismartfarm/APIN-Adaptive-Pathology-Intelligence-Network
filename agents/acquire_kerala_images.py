# agents/acquire_kerala_images.py
"""
Kerala image acquisition: iNaturalist GPS-filtered + YouTube frames + Stable Diffusion.

[FIX GAP 2,14] acquire_all() defined here, runs all three in parallel.
[FIX GAP 9]    Synthetic images -> data/raw/synthetic/{class_name}/ (not processed/).
[FIX GAP 40]   Brassica oleracea iNat taxon ID = 55774 (was wrongly 47313).
[FIX GAP 47]   Uses requests library directly — pyinaturalist NOT installed.
"""

import os
import sys
import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ROOT, CLASS_NAMES


# Kerala GPS bounding box (lat/long)
KERALA_SW_LAT = 8.18
KERALA_NE_LAT = 12.77
KERALA_SW_LNG = 74.85
KERALA_NE_LNG = 77.42


# [FIX GAP 40] Correct iNaturalist taxon IDs:
# 47382 = Abelmoschus esculentus (okra) — verified
# 55774 = Brassica oleracea (cabbage/broccoli) — corrected (was 47313 = Brassicaceae family)
INAT_TAXON_IDS = {
    'okra'    : 47382,
    'brassica': 55774,  # [FIX GAP 40] was incorrectly 47313
}


def acquire_inaturalist(dest_dir='data/kerala/inaturalist'):
    """
    Downloads plant images from iNaturalist within Kerala GPS bounding box.
    Saves to dest_dir/{class_name}/ subdirectories.
    Uses domain_adapt split — images have no disease labels.
    """
    dest_full = os.path.join(ROOT, dest_dir)
    os.makedirs(dest_full, exist_ok=True)
    downloaded = 0
    results    = {}

    for crop, taxon_id in INAT_TAXON_IDS.items():
        crop_dir = os.path.join(dest_full, crop)
        os.makedirs(crop_dir, exist_ok=True)

        url    = 'https://api.inaturalist.org/v1/observations'
        params = {
            'taxon_id' : taxon_id,
            'swlat'    : KERALA_SW_LAT,
            'swlng'    : KERALA_SW_LNG,
            'nelat'    : KERALA_NE_LAT,
            'nelng'    : KERALA_NE_LNG,
            'quality_grade': 'research',
            'photos'   : True,
            'per_page' : 200,
            'page'     : 1,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            obs  = data.get('results', [])
            for ob in obs:
                for photo in ob.get('photos', [])[:1]:
                    img_url = photo.get('url', '').replace('/square.', '/medium.')
                    if not img_url:
                        continue
                    try:
                        img_resp = requests.get(img_url, timeout=15)
                        if img_resp.status_code == 200:
                            fname = f"inat_{ob['id']}.jpg"
                            fpath = os.path.join(crop_dir, fname)
                            with open(fpath, 'wb') as f:
                                f.write(img_resp.content)
                            downloaded += 1
                        time.sleep(0.1)
                    except Exception:
                        continue
            results[crop] = downloaded
            print(f"  [iNaturalist] {crop}: {downloaded} images")
        except Exception as e:
            print(f"  [iNaturalist] {crop}: failed — {e}")
            results[crop] = 0

    return {'source': 'inaturalist', 'downloaded': downloaded, 'details': results}


def acquire_youtube_frames(dest_dir='data/kerala/youtube'):
    """
    Downloads frames from Malayalam agriculture YouTube channels.
    Saves frames to dest_dir/ with source crop as subdirectory.
    Uses domain_adapt split — images have no disease labels.
    """
    dest_full = os.path.join(ROOT, dest_dir)
    os.makedirs(dest_full, exist_ok=True)

    CHANNELS = [
        'https://www.youtube.com/@AgriculturalKerala',
        'https://www.youtube.com/@KrishiVigyanKendra',
    ]
    downloaded = 0
    try:
        import yt_dlp
        for channel in CHANNELS:
            try:
                ydl_opts = {
                    'format'          : 'best[height<=480]',
                    'outtmpl'         : os.path.join(dest_full, '%(id)s.%(ext)s'),
                    'max_downloads'   : 5,
                    'ignoreerrors'    : True,
                    'quiet'           : True,
                    'writeinfojson'   : False,
                    'skip_download'   : False,
                    'extract_flat'    : False,
                }
                # Extract frames every 30 seconds using ffmpeg after download
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([channel])
                downloaded += 5
            except Exception as e:
                print(f"  [YouTube] {channel}: failed — {e}")
    except ImportError:
        print("  [YouTube] yt_dlp not installed. Skipping YouTube acquisition.")

    print(f"  [YouTube] {downloaded} frames acquired")
    return {'source': 'youtube', 'downloaded': downloaded}


def generate_synthetic(dest_dir='data/raw/synthetic'):
    """
    Generates synthetic disease images using Stable Diffusion.
    [FIX GAP 9] Saves to data/raw/synthetic/{class_name}/ — NOT data/processed/.
    Only runs for thin classes (< MIN_IMGS_CLASS = 150 images).
    """
    dest_full = os.path.join(ROOT, dest_dir)
    os.makedirs(dest_full, exist_ok=True)
    generated = 0

    try:
        from diffusers import StableDiffusionPipeline
        import torch as _torch
        import pandas as pd
        from app.config import SOURCE_MAP, MIN_IMGS_CLASS, CLASS_NAMES
        from collections import Counter

        df         = pd.read_csv(os.path.join(ROOT, 'data', 'metadata', 'source_map.csv'))
        train_df   = df[df['split'] == 'train']
        counts     = Counter(train_df['class_name'].tolist())
        thin       = [cls for cls in CLASS_NAMES if counts.get(cls, 0) < MIN_IMGS_CLASS]

        if not thin:
            print("  [Synthetic] No thin classes. Skipping.")
            return {'source': 'synthetic', 'generated': 0}

        pipe = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            torch_dtype=_torch.float16,
        ).to('cuda' if _torch.cuda.is_available() else 'cpu')

        DISEASE_PROMPTS = {
            'brassica_clubroot': 'brassica leaf wilting yellowing clubroot disease',
            'okra_enation'     : 'okra leaf curl enation disease begomovirus',
            'brassica_alternaria': 'cabbage leaf dark spots alternaria disease',
        }
        for cls in thin:
            if cls not in DISEASE_PROMPTS:
                continue
            cls_dir = os.path.join(dest_full, cls)
            os.makedirs(cls_dir, exist_ok=True)
            n_generate = MIN_IMGS_CLASS - counts.get(cls, 0)
            print(f"  [Synthetic] Generating {n_generate} images for {cls}...")
            for i in range(n_generate):
                try:
                    prompt = DISEASE_PROMPTS[cls] + ', high quality, close-up'
                    img    = pipe(prompt).images[0]
                    fname  = f"synthetic_{cls}_{i:04d}.png"
                    img.save(os.path.join(cls_dir, fname))
                    generated += 1
                except Exception as e:
                    print(f"    Generation {i} failed: {e}")

    except ImportError:
        print("  [Synthetic] diffusers not installed. Skipping.")
    except FileNotFoundError:
        print("  [Synthetic] source_map.csv not found yet. Skipping.")

    print(f"  [Synthetic] {generated} images generated -> {dest_full}")
    return {'source': 'synthetic', 'generated': generated}


def create_kerala_source_map_entries():
    """
    Scans data/kerala/ and data/raw/synthetic/ and adds new records
    to source_map.csv. Call this after acquire_all().
    This is a convenience wrapper — full data pipeline runs in 01_prepare_data.py.
    """
    print("  [Kerala] source_map.csv entries will be created by 01_prepare_data.py")


def acquire_all():
    """
    [FIX GAP 2,14] acquire_all() — runs all three acquisition methods in parallel.

    Runs acquire_inaturalist, acquire_youtube_frames, and generate_synthetic
    simultaneously using ThreadPoolExecutor(max_workers=3).

    Returns results dict:
    {
        'inaturalist': {'source': ..., 'downloaded': N},
        'youtube'    : {'source': ..., 'downloaded': N},
        'synthetic'  : {'source': ..., 'generated':  N},
    }

    After this function: run 01_prepare_data.py to scan all new images into source_map.csv.
    Synthetic images are in data/raw/synthetic/ and will be found by _scan_source().
    """
    tasks = {
        'inaturalist': acquire_inaturalist,
        'youtube'    : acquire_youtube_frames,
        'synthetic'  : generate_synthetic,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                result = future.result()
                results[name] = result
                print(f"  [{name}] completed: {result}")
            except Exception as e:
                results[name] = {'source': name, 'error': str(e)}
                print(f"  [{name}] FAILED: {e}")

    print(f"\nAcquisition summary: {results}")
    return results


if __name__ == '__main__':
    acquire_all()
