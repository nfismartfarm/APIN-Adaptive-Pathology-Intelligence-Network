# setup/package_deployment.py
"""Step 14: Creates Dockerfile and verifies deployment files are present."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ROOT


def package():
    print("=" * 50)
    print("STEP 14 — PACKAGE DEPLOYMENT")
    print("=" * 50)

    dockerfile = os.path.join(ROOT, 'Dockerfile')
    content = '''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/best_model.pt models/best_model.pt
COPY models/temperature.pt models/temperature.pt
COPY diagnosis/ diagnosis/
COPY templates/ templates/
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
'''
    with open(dockerfile, 'w') as f:
        f.write(content)
    print(f"  Dockerfile written to {dockerfile}")

    # Verify deployment files
    required = [
        'app/config.py', 'app/model.py', 'app/inference.py',
        'app/validator.py', 'app/main.py',
        'diagnosis/diagnosis_lookup.json',
        'templates/index.html', 'static/style.css', 'static/app.js',
        'requirements.txt',
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        print(f"  WARNING: Missing deployment files: {missing}")
    else:
        print(f"  All {len(required)} deployment files present.")

    print("\nStep 14 complete.")


if __name__ == '__main__':
    package()
