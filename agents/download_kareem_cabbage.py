# agents/download_kareem_cabbage.py
"""Downloads Kareem's Cabbage Disease dataset from Kaggle."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.kaggle_utils import kaggle_download
from app.config import RAW


SLUG     = 'kareemabdulkareemsabbir/cabbage-disease-dataset'
DEST_DIR = os.path.join(RAW, 'kareem_cabbage')
MIN_IMGS = 3800


def download():
    print(f"[kareem_cabbage] Downloading from {SLUG}...")
    try:
        kaggle_download(SLUG, DEST_DIR)
        count = sum(1 for r, d, f in os.walk(DEST_DIR)
                    for fn in f if os.path.splitext(fn)[1].lower() in {'.jpg','.jpeg','.png','.webp'})
        print(f"[kareem_cabbage] {count} images downloaded.")
        return {'name': 'kareem_cabbage', 'success': True, 'count': count}
    except Exception as e:
        print(f"[kareem_cabbage] FAILED: {e}")
        return {'name': 'kareem_cabbage', 'success': False, 'error': str(e)}


if __name__ == '__main__':
    download()
