# agents/download_iubat_okra.py
"""Downloads IUBAT Okra Disease dataset. Falls back to manual download instructions."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import RAW


DEST_DIR = os.path.join(RAW, 'iubat_okra')
# Roboflow dataset — may require manual download
ROBOFLOW_URL = 'https://universe.roboflow.com/iubat/okra-disease-detection'
MIN_IMGS = 2000


def download():
    print(f"[iubat_okra] Attempting download...")
    os.makedirs(DEST_DIR, exist_ok=True)

    # Check if already downloaded manually
    count = sum(1 for r, d, f in os.walk(DEST_DIR)
                for fn in f if os.path.splitext(fn)[1].lower() in {'.jpg','.jpeg','.png','.webp'})
    if count >= MIN_IMGS:
        print(f"[iubat_okra] Already have {count} images. Skipping download.")
        return {'name': 'iubat_okra', 'success': True, 'count': count}

    # Try Kaggle first
    try:
        from agents.kaggle_utils import kaggle_download
        kaggle_download('iubat/okra-disease-detection', DEST_DIR)
        count = sum(1 for r, d, f in os.walk(DEST_DIR)
                    for fn in f if os.path.splitext(fn)[1].lower() in {'.jpg','.jpeg','.png','.webp'})
        if count >= MIN_IMGS:
            return {'name': 'iubat_okra', 'success': True, 'count': count}
    except Exception as e:
        print(f"[iubat_okra] Kaggle download failed: {e}")

    # Write manual download instructions
    manual_path = os.path.join(DEST_DIR, 'MANUAL_DOWNLOAD_REQUIRED.txt')
    with open(manual_path, 'w') as f:
        f.write(f"IUBAT Okra Disease Dataset requires manual download.\n")
        f.write(f"URL: {ROBOFLOW_URL}\n")
        f.write(f"Download as 'Folder Structure' format.\n")
        f.write(f"Unzip into: {DEST_DIR}\n")
    print(f"[iubat_okra] Manual download required. See {manual_path}")
    return {'name': 'iubat_okra', 'success': False, 'error': 'Manual download required'}


if __name__ == '__main__':
    download()
