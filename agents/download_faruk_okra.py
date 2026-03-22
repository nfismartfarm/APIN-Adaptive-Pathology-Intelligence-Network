# agents/download_faruk_okra.py
"""Downloads Faruk's Okra Disease dataset from Kaggle."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.kaggle_utils import kaggle_download
from app.config import RAW


SLUG     = 'farukalam/okra-leaf-disease'
DEST_DIR = os.path.join(RAW, 'faruk_okra')
MIN_IMGS = 1600


def download():
    print(f"[faruk_okra] Downloading from {SLUG}...")
    try:
        kaggle_download(SLUG, DEST_DIR)
        count = sum(1 for r, d, f in os.walk(DEST_DIR)
                    for fn in f if os.path.splitext(fn)[1].lower() in {'.jpg','.jpeg','.png','.webp'})
        print(f"[faruk_okra] {count} images downloaded.")
        return {'name': 'faruk_okra', 'success': True, 'count': count}
    except Exception as e:
        print(f"[faruk_okra] FAILED: {e}")
        return {'name': 'faruk_okra', 'success': False, 'error': str(e)}


if __name__ == '__main__':
    download()
