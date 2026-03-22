# agents/download_plantdoc.py
"""Downloads PlantDoc dataset from Kaggle. Tier-2 test only — NEVER training."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.kaggle_utils import kaggle_download
from app.config import PLANTDOC_DIR


SLUG     = 'pratikkayal/plantdoc-dataset'
MIN_IMGS = 2000


def download():
    print(f"[plantdoc] Downloading from {SLUG}...")
    try:
        kaggle_download(SLUG, PLANTDOC_DIR)
        count = sum(1 for r, d, f in os.walk(PLANTDOC_DIR)
                    for fn in f if os.path.splitext(fn)[1].lower() in {'.jpg','.jpeg','.png','.webp'})
        print(f"[plantdoc] {count} images downloaded (TIER-2 TEST ONLY).")
        return {'name': 'plantdoc', 'success': True, 'count': count}
    except Exception as e:
        print(f"[plantdoc] FAILED: {e}")
        return {'name': 'plantdoc', 'success': False, 'error': str(e)}


if __name__ == '__main__':
    download()
