# setup/smoke_tests.py
"""Collection of smoke test functions for pipeline steps."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all critical packages can be imported."""
    modules = [
        'torch', 'torchvision', 'timm', 'albumentations',
        'sklearn', 'cv2', 'fastapi', 'wandb', 'pandas',
    ]
    results = {}
    for mod in modules:
        try:
            __import__(mod)
            results[mod] = True
        except ImportError:
            results[mod] = False
    return results


def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        return {
            'available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'cuda_version': torch.version.cuda or 'N/A',
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


def test_model_creation():
    """Test that the model can be instantiated."""
    try:
        from app.model import PlantDiseaseModel
        model = PlantDiseaseModel()
        param_count = sum(p.numel() for p in model.parameters())
        return {'success': True, 'param_count': param_count}
    except Exception as e:
        return {'success': False, 'error': str(e)}


if __name__ == '__main__':
    print("Import tests:", test_imports())
    print("CUDA test:", test_cuda())
    print("Model test:", test_model_creation())
