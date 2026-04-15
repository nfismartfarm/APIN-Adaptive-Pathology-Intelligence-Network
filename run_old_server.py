"""Run the original 10-class EfficientNetV2-S server on port 8001."""
import sys, os

# Point Python to old_10class as the app package
OLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'old_10class')
sys.path.insert(0, OLD_DIR)

# Override ROOT in the old config to point to old_10class
os.environ['_OLD_SERVER_ROOT'] = OLD_DIR

# Patch the old config module before anything imports it
import importlib
import app.config as cfg

# Fix paths to point to old_10class directory
cfg.ROOT = OLD_DIR
cfg.BEST_MODEL = os.path.join(OLD_DIR, 'models', 'best_model.pt')
cfg.DIAG_JSON = os.path.join(OLD_DIR, 'diagnosis', 'diagnosis_lookup.json')
cfg.TEMP_PATH = os.path.join(OLD_DIR, 'models', 'temperature.pt')
cfg.DEVICE = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=False)
