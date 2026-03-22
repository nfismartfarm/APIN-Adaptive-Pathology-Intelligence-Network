# setup/setup_project.py
"""
Step 0: Creates project directories, loads .env, validates env vars,
configures GitHub.
[FIX GAP 59] Checks that a virtual environment is active.
[FIX GAP 65] Loads .env at startup.
"""

import os
import sys
import subprocess

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_venv():
    """[FIX GAP 59] Verify running inside a virtual environment."""
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    if not in_venv:
        print("ERROR: Not running inside a virtual environment.")
        print("Create and activate one first:")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate.bat   (Windows CMD)")
        print("  venv\\Scripts\\Activate.ps1   (Windows PowerShell)")
        sys.exit(1)
    print(f"  Virtual environment active: {sys.prefix}")


def create_directories():
    """Create all required project directories."""
    dirs = [
        'data/raw', 'data/processed/train', 'data/processed/val',
        'data/processed/test', 'data/metadata', 'data/kerala', 'data/plantdoc',
        'models/checkpoints', 'cache', 'reports', 'diagnosis', 'tools',
        'static', 'templates', 'agents', 'training', 'app', 'setup',
    ]
    for d in dirs:
        path = os.path.join(ROOT, d)
        os.makedirs(path, exist_ok=True)
    print(f"  Created {len(dirs)} directories.")


def check_env_vars():
    """Validate required environment variables are set."""
    required = ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'GITHUB_TOKEN']
    optional = ['WANDB_API_KEY']
    missing  = []
    for var in required:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"ERROR: Missing required env vars: {missing}")
        print(f"Copy .env.template to .env and fill in values.")
        sys.exit(1)
    for var in optional:
        if not os.environ.get(var):
            print(f"  WARNING: {var} not set. wandb will run offline.")
    print(f"  Environment variables validated.")


def configure_git():
    """Configure git remote and initial commit if needed."""
    github_token = os.environ.get('GITHUB_TOKEN', '')
    github_repo  = os.environ.get('GITHUB_REPO', '')
    if not github_repo:
        print("  GITHUB_REPO not set — skipping git configuration.")
        return
    r = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        remote_url = f'https://{github_token}@github.com/{github_repo}.git'
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url],
                       cwd=ROOT)
        print(f"  Git remote set to {github_repo}")
    else:
        print(f"  Git remote already configured.")


def write_gitignore():
    """Write .gitignore if not present."""
    path = os.path.join(ROOT, '.gitignore')
    if os.path.exists(path):
        return
    content = """
# Data (gigabytes — never commit)
data/raw/
data/processed/
data/kerala/
data/plantdoc/
data/metadata/*.csv
cache/

# Model weights
models/*.pt
models/checkpoints/

# Environment
.env
venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Feedback database
feedback.db

# Pipeline state
.pipeline_progress.json
pipeline_failures.log
""".strip()
    with open(path, 'w') as f:
        f.write(content)
    print(f"  .gitignore written.")


if __name__ == '__main__':
    print("=" * 50)
    print("STEP 00 — PROJECT SETUP")
    print("=" * 50)
    check_venv()
    create_directories()
    check_env_vars()
    configure_git()
    write_gitignore()
    print("\nStep 00 complete.")
