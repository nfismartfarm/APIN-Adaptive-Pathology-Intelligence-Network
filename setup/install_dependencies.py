# setup/install_dependencies.py
"""Step 2: Install all Python dependencies from requirements_train.txt."""

import os
import sys
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def install():
    print("=" * 50)
    print("STEP 02 — INSTALL DEPENDENCIES")
    print("=" * 50)

    req_file = os.path.join(ROOT, 'requirements_train.txt')
    if not os.path.exists(req_file):
        print(f"ERROR: {req_file} not found.")
        sys.exit(1)

    print(f"Installing from {req_file}...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', req_file],
        capture_output=False,
    )
    if result.returncode != 0:
        print("WARNING: Some packages may have failed to install.")
        print("Check output above for errors.")

    # Verify critical imports
    print("\nVerifying critical imports...")
    critical = [
        'torch', 'torchvision', 'timm', 'albumentations',
        'sklearn', 'cv2', 'fastapi', 'wandb', 'pandas', 'PIL',
    ]
    failed = []
    for mod in critical:
        try:
            __import__(mod)
            print(f"  OK: {mod}")
        except ImportError:
            print(f"  FAIL: {mod}")
            failed.append(mod)

    if failed:
        print(f"\nFailed imports: {failed}")
        print("Install them manually before proceeding.")
    else:
        print("\nAll critical imports verified.")

    print("\nStep 02 complete.")


if __name__ == '__main__':
    install()
