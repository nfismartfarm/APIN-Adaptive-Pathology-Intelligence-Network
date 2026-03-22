# setup/test_server.py
"""
[FIX GAP 5] Test image is a textured JPEG (noise overlay on green base).
A solid-colour rectangle has Laplacian variance = 0 → fails blur check.
Noise overlay gives variance >> 80 → passes.
"""

import sys
import io
import requests
import numpy as np
from PIL import Image

SERVER_URL = 'http://localhost:8765'


def create_textured_test_image():
    """Green rectangle + heavy random noise. Passes blur check."""
    rng  = np.random.default_rng(seed=42)
    base = np.zeros((300, 300, 3), dtype=np.uint8)
    base[:, :, 1] = 120   # green
    noise = rng.integers(0, 60, (300, 300, 3), dtype=np.uint8)
    img   = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    buf   = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def run_smoke_test():
    print("Running server smoke test...")

    try:
        r = requests.get(f'{SERVER_URL}/health', timeout=5)
        assert r.status_code == 200, f"Health returned {r.status_code}"
        print(f"  ✓ Health: {r.json()}")
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        sys.exit(1)

    try:
        img_bytes = create_textured_test_image()
        r = requests.post(
            f'{SERVER_URL}/predict',
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            timeout=60,
        )
        assert r.status_code == 200, f"Predict returned {r.status_code}: {r.text[:300]}"
        result = r.json()
        required = ['crop', 'diseases', 'confidence', 'uncertainty',
                    'severity', 'treatment', 'urgency', 'ood_flagged']
        for key in required:
            assert key in result, f"Missing key: {key}"
        print(f"  ✓ Predict: crop={result['crop']} diseases={result['diseases']}")
    except Exception as e:
        print(f"  ✗ Predict test failed: {e}")
        sys.exit(1)

    print("Server smoke test PASSED")
    sys.exit(0)


if __name__ == '__main__':
    run_smoke_test()
