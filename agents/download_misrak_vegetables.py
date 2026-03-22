# agents/download_misrak_vegetables.py
"""Downloads Misrak's Vegetable dataset from Kaggle. Only brassica classes kept."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.kaggle_utils import kaggle_download
from app.config import RAW


SLUG     = 'misaborishade/vegetable-disease-recognition-dataset'
DEST_DIR = os.path.join(RAW, 'misrak_veg')
MIN_IMGS = 1500


def download():
    print(f"[misrak_veg] Downloading from {SLUG}...")
    try:
        kaggle_download(SLUG, DEST_DIR)
        count = sum(1 for r, d, f in os.walk(DEST_DIR)
                    for fn in f if os.path.splitext(fn)[1].lower() in {'.jpg','.jpeg','.png','.webp'})
        print(f"[misrak_veg] {count} total images (brassica filtered in 01_prepare_data.py).")
        return {'name': 'misrak_veg', 'success': True, 'count': count}
    except Exception as e:
        print(f"[misrak_veg] FAILED: {e}")
        return {'name': 'misrak_veg', 'success': False, 'error': str(e)}


if __name__ == '__main__':
    download()
