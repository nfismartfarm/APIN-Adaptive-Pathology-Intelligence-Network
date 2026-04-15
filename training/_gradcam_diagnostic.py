"""Grad-CAM diagnostic on 3 PlantDoc images: late_blight, septoria, bacterial_spot."""
import torch, numpy as np, pandas as pd, os, sys, base64, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from app.config import ROOT, DEVICE, BEST_MODEL, CLASS_NAMES, DISEASE_THRESHOLDS
from app.model import load_model_for_inference
from app.inference import run_inference

print("Loading model...")
model = load_model_for_inference(os.path.join(ROOT, BEST_MODEL), DEVICE)
model.eval()
print("Model loaded.")

df = pd.read_csv("data/metadata/source_map.csv")
pd_eval = df[df["source_dataset"].str.contains("plantdoc_eval", case=False, na=False)]

target_classes = [
    "tomato_late_blight",        # Strong (F1=0.779)
    "tomato_septoria_leaf_spot", # Strong (F1=0.675)
    "tomato_bacterial_spot",     # Weak (F1=0.311)
]

plots_dir = os.path.join(ROOT, "data_prep", "plots")
os.makedirs(plots_dir, exist_ok=True)

for cls_name in target_classes:
    cls_images = pd_eval[pd_eval["class_name"] == cls_name]
    if len(cls_images) == 0:
        print(f"No images found for {cls_name}")
        continue

    img_path = cls_images.iloc[0]["image_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT, img_path)

    img_pil = Image.open(img_path).convert("RGB")
    # Resize large images to max 800px to avoid CUDA OOM on SAM
    max_dim = 800
    if max(img_pil.size) > max_dim:
        ratio = max_dim / max(img_pil.size)
        new_size = (int(img_pil.size[0] * ratio), int(img_pil.size[1] * ratio))
        img_pil = img_pil.resize(new_size, Image.LANCZOS)
    img_np = np.array(img_pil)

    # Clear CUDA cache before each inference to prevent OOM accumulation
    import torch as _t
    _t.cuda.empty_cache()

    result = run_inference(model, img_np)

    print(f"Class: {cls_name}")
    print(f"  Image size: {img_np.shape}")
    print(f"  Predicted crop: {result['crop']}")
    print(f"  Diseases: {result['diseases']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Uncertainty: {result['uncertainty']:.3f}")
    print(f"  OOD flagged: {result['ood_flagged']}")

    # Decode and analyse heatmap
    if result["heatmap_b64"]:
        heatmap_bytes = base64.b64decode(result["heatmap_b64"])
        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
        heatmap_np = np.array(heatmap_img)

        # Convert to grayscale for analysis
        if heatmap_np.ndim == 3:
            heatmap_gray = heatmap_np.mean(axis=2)
        else:
            heatmap_gray = heatmap_np.astype(float)

        hmap_min, hmap_max = heatmap_gray.min(), heatmap_gray.max()
        if hmap_max - hmap_min > 1e-6:
            heatmap_norm = (heatmap_gray - hmap_min) / (hmap_max - hmap_min)
        else:
            heatmap_norm = np.zeros_like(heatmap_gray)

        high_act_fraction = (heatmap_norm > 0.5).mean()
        print(f"  Heatmap high-activation coverage: {high_act_fraction:.1%}")
        if high_act_fraction < 0.10:
            print(f"  --> Very focused (excellent for 2nd pass)")
        elif high_act_fraction < 0.30:
            print(f"  --> Moderately focused (good for 2nd pass)")
        elif high_act_fraction < 0.60:
            print(f"  --> Somewhat diffuse (2nd pass marginal)")
        else:
            print(f"  --> Covers most of image (2nd pass will NOT help)")

        # Find bounding box of activation
        high_mask = heatmap_norm > 0.5
        if high_mask.any():
            rows = np.any(high_mask, axis=1)
            cols = np.any(high_mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            print(f"  Activation bbox: ({x1},{y1}) to ({x2},{y2}), size {bbox_w}x{bbox_h}")
        else:
            print(f"  No high activation region found")

        # Save diagnostic plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_pil)
        axes[0].set_title(f"Original: {cls_name}")
        axes[0].axis("off")

        axes[1].imshow(heatmap_img)
        axes[1].set_title("Grad-CAM heatmap")
        axes[1].axis("off")

        axes[2].imshow(img_pil)
        axes[2].imshow(heatmap_img, alpha=0.5)
        axes[2].set_title(f"Overlay | coverage={high_act_fraction:.1%}")
        axes[2].axis("off")

        safe_name = cls_name.replace("/", "_")
        out_path = os.path.join(plots_dir, f"gradcam_diagnostic_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved to: {out_path}")
    else:
        print(f"  No heatmap generated (OOD or error)")

    print()
