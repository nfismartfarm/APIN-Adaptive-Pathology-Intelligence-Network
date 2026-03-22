# agents/kaggle_utils.py
"""
Shared Kaggle credential setup.
All Kaggle download agents import from here.
Handles both old and new (KGAT_) Kaggle API token formats.
"""

import os
import json
import subprocess
import sys


def setup_kaggle_credentials():
    """
    Reads KAGGLE_USERNAME and KAGGLE_KEY from environment variables.
    Writes ~/.kaggle/kaggle.json if not already present.
    Also sets KAGGLE_API_TOKEN for new kaggle CLI that reads it directly.
    Raises EnvironmentError if credentials not found.
    """
    username = os.environ.get('KAGGLE_USERNAME')
    key      = os.environ.get('KAGGLE_KEY')
    if not username or not key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in environment/.env.\n"
            "Get them from kaggle.com > Account > Create API Token."
        )
    # Set KAGGLE_API_TOKEN for new kaggle CLI (KGAT_ token format)
    os.environ["KAGGLE_API_TOKEN"] = key
    kaggle_dir  = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json):
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_json, 'w') as f:
            json.dump({'username': username, 'key': key}, f)
        # On Windows, os.chmod with 0o600 may not work the same but is harmless
        try:
            os.chmod(kaggle_json, 0o600)
        except OSError:
            pass
        print("  Kaggle credentials written to ~/.kaggle/kaggle.json")


def kaggle_download(slug, dest_dir):
    """
    Downloads and unzips a Kaggle dataset.
    slug: 'owner/dataset-name'
    dest_dir: destination directory (created if missing)
    """
    setup_kaggle_credentials()
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'kaggle', 'datasets', 'download',
        '-d', slug,
        '--path', dest_dir,
        '--unzip',
        '--quiet',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for {slug}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print(f"  Downloaded and unzipped: {slug} -> {dest_dir}")
