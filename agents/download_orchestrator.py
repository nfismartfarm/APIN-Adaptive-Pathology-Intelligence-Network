# agents/download_orchestrator.py
"""Runs all dataset downloads in parallel using ThreadPoolExecutor."""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import META

# Load .env for Kaggle credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_all_downloads():
    """Downloads all datasets in parallel. Returns dict of results."""
    from agents.download_sabbir_okra import download as dl_sabbir
    from agents.download_iubat_okra import download as dl_iubat
    from agents.download_kareem_cabbage import download as dl_kareem
    from agents.download_misrak_vegetables import download as dl_misrak
    from agents.download_faruk_okra import download as dl_faruk
    from agents.download_ghose_cabbage import download as dl_ghose
    from agents.download_plantdoc import download as dl_plantdoc

    tasks = {
        'sabbir_okra'   : dl_sabbir,
        'iubat_okra'    : dl_iubat,
        'kareem_cabbage': dl_kareem,
        'misrak_veg'    : dl_misrak,
        'faruk_okra'    : dl_faruk,
        'ghose_cabbage' : dl_ghose,
        'plantdoc'      : dl_plantdoc,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                result = future.result()
                results[name] = result
            except Exception as e:
                results[name] = {'name': name, 'success': False, 'error': str(e)}

    # Write results to metadata
    os.makedirs(META, exist_ok=True)
    results_path = os.path.join(META, 'download_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDownload results saved to {results_path}")

    # Print summary
    for name, r in results.items():
        status = 'OK' if r.get('success') else 'FAIL'
        count  = r.get('count', 0)
        err    = r.get('error', '')
        print(f"  [{status}] {name}: {count} images  {err}")

    return results


if __name__ == '__main__':
    run_all_downloads()
