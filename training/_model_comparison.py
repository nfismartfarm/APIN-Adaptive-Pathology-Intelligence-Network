"""Print detailed comparison of EfficientNetV2-S vs Swin-Tiny."""

print("=" * 80)
print("DETAILED COMPARISON: EfficientNetV2-S (23-class) vs Swin-Tiny (23-class)")
print("=" * 80)
print()

print("MODEL SPECIFICATIONS")
print("-" * 80)
rows = [
    ("Backbone",             "tf_efficientnetv2_s",  "swin_tiny_224"),
    ("Parameters",           "~22M",                 "28.7M"),
    ("Model file size",      "84.2MB",               "114.9MB"),
    ("FPN channels",         "[64,160,256]",          "[192,384,768]"),
    ("Pooling",              "Global Avg Pool",       "Attention Pool"),
    ("Crop conditioning",    "FiLM",                  "Cond Layer Norm"),
    ("Disease head",         "Unified 23-class",      "MoE 4-expert"),
    ("Distillation",         "None",                  "DeiT token"),
    ("Multi-scale inference","Yes (192,224,288)",      "No (224 only)"),
]
print(f"  {'':.<43} {'EfficientNetV2-S':>17} {'Swin-Tiny':>17}")
for label, v1, v2 in rows:
    print(f"  {label:.<43} {v1:>17} {v2:>17}")
print()

print("TRAINING DETAILS")
print("-" * 80)
rows = [
    ("Training images",      "~48,000",              "14,437"),
    ("Crop ratio",           "13:1",                 "2.25:1"),
    ("Loss function",        "BCE + CE",             "Focal+CORAL+Arc"),
    ("Domain adaptation",    "None",                 "CORAL (0.01)"),
    ("Distillation",         "None",                 "DeiT (a=0.5,T=3)"),
    ("Metric learning",      "None",                 "ArcFace (0.1)"),
    ("Data augmentation",    "Albumentations",       "CutMix+RandAug"),
    ("LR schedule",          "OneCycleLR",           "CosineWarmRestart"),
    ("Phase 1 (heads)",      "~30 min",              "3.7 hr"),
    ("Phase 2 (full)",       "~50 min",              "2.6 hr"),
]
print(f"  {'':.<43} {'EfficientNetV2-S':>17} {'Swin-Tiny':>17}")
for label, v1, v2 in rows:
    print(f"  {label:.<43} {v1:>17} {v2:>17}")
print()

print("CONTROLLED TEST SET RESULTS")
print("-" * 80)
rows = [
    ("Macro F1",             "0.9523",               "0.9410"),
    ("Crop accuracy",        "~99%",                 "99.55%"),
    ("ECE (calibration)",    "unknown",              "0.068"),
]
print(f"  {'':.<43} {'EfficientNetV2-S':>17} {'Swin-Tiny':>17}")
for label, v1, v2 in rows:
    print(f"  {label:.<43} {v1:>17} {v2:>17}")
print()

print("PLANTDOC WILD-CONDITION RESULTS (7 tomato classes)")
print("-" * 80)
rows = [
    ("Macro F1 (default 0.50)",  "unknown*",         "0.4415"),
    ("Macro F1 (tuned thresh)",  "N/A",              "0.5630"),
]
print(f"  {'':.<43} {'EfficientNetV2-S':>17} {'Swin-Tiny':>17}")
for label, v1, v2 in rows:
    print(f"  {label:.<43} {v1:>17} {v2:>17}")
print("  * Old PlantDoc eval was 4 brassica classes only, not comparable")
print()

print("PLANTDOC PER-CLASS F1 (Swin-Tiny, per-class thresholds)")
print("-" * 80)
pd_classes = [
    ("tomato_late_blight",            0.779, 0.300),
    ("tomato_yellow_leaf_curl_virus", 0.724, 0.375),
    ("tomato_septoria_leaf_spot",     0.675, 0.300),
    ("tomato_leaf_mold",              0.522, 0.325),
    ("tomato_early_blight",           0.505, 0.600),
    ("tomato_mosaic_virus",           0.425, 0.400),
    ("tomato_bacterial_spot",         0.311, 0.175),
]
print(f"  {'Class':<42} {'F1':>6} {'Thresh':>8}")
for cls, f1, t in pd_classes:
    print(f"  {cls:<42} {f1:>6.3f} {t:>8.3f}")
print()

print("DOMAIN GAP (Swin-Tiny: Controlled vs PlantDoc)")
print("-" * 80)
gap_data = [
    ("tomato_mosaic_virus",           0.979, 0.434, 0.546),
    ("tomato_early_blight",           0.870, 0.505, 0.365),
    ("tomato_leaf_mold",              0.882, 0.522, 0.360),
    ("tomato_bacterial_spot",         0.681, 0.325, 0.356),
    ("tomato_yellow_leaf_curl_virus", 0.969, 0.692, 0.276),
    ("tomato_septoria_leaf_spot",     0.829, 0.656, 0.173),
    ("tomato_late_blight",            0.881, 0.784, 0.096),
    ("MACRO",                         0.870, 0.560, 0.310),
]
print(f"  {'Class':<42} {'Ctrl':>7} {'PD':>7} {'Gap':>7}")
for cls, ctrl, pd, gap in gap_data:
    print(f"  {cls:<42} {ctrl:>7.3f} {pd:>7.3f} {gap:>7.3f}")
print()

print("COMPONENT ABLATION (Swin-Tiny, 500 val images)")
print("-" * 80)
abl_data = [
    ("Full model (baseline)",               0.8626, "---"),
    ("Without Attention Pooling (GAP)",      0.1099, "-0.753"),
    ("Without Cross-crop masking",           0.6795, "-0.183"),
    ("Without MoE routing (uniform)",        0.7558, "-0.107"),
    ("Without CLN (identity)",               0.7680, "-0.095"),
]
print(f"  {'Configuration':<45} {'Val F1':>8} {'Drop':>8}")
for cfg, f1, drop in abl_data:
    print(f"  {cfg:<45} {f1:>8.4f} {drop:>8}")
print()

print("CROP CLASSIFICATION ON REAL-WORLD iNATURALIST PHOTOS")
print("-" * 80)
crop_data = [
    ("Overall accuracy",     "unknown",     "94.0% (126/134)"),
    ("Okra",                 "unknown",     "93.8% (30/32)"),
    ("Brassica",             "unknown",     "92.3% (12/13)"),
    ("Tomato",               "unknown",     "94.9% (75/79)"),
    ("Chilli",               "0% (broken)", "90.0% (9/10)"),
]
print(f"  {'':.<43} {'EfficientNetV2-S':>17} {'Swin-Tiny':>17}")
for label, v1, v2 in crop_data:
    print(f"  {label:.<43} {v1:>17} {v2:>17}")
print()

print("INFERENCE PIPELINE")
print("-" * 80)
inf_data = [
    ("Inference time",       "~5-10s",              "~1.7s"),
    ("MC Dropout passes",    "5",                   "5"),
    ("MobileSAM",            "Yes",                 "Yes"),
    ("TTA (flip)",           "Yes",                 "Yes"),
    ("Multi-scale",          "Yes (3 scales)",      "No (224 only)"),
    ("Grad-CAM 2nd pass",    "N/A",                 "Disabled (hurts)"),
    ("Per-class thresholds", "No (0.50 all)",       "Yes (tuned)"),
    ("Soft cross-crop mask", "No",                  "Yes (low conf)"),
]
print(f"  {'':.<43} {'EfficientNetV2-S':>17} {'Swin-Tiny':>17}")
for label, v1, v2 in inf_data:
    print(f"  {label:.<43} {v1:>17} {v2:>17}")
print()

print("PHASE 3 INFERENCE IMPROVEMENTS (Swin-Tiny)")
print("-" * 80)
p3_data = [
    ("Per-class threshold tuning",   "0.4415 -> 0.5630 (+0.122)"),
    ("Grad-CAM 2nd pass (0.60)",     "0.5630 -> 0.4068 (HURT -0.156)"),
    ("Grad-CAM 2nd pass (0.10)",     "0.5630 -> 0.4130 (HURT -0.150)"),
    ("Uncertainty ensemble",         "Never triggered (unc always < 0.30)"),
    ("Soft cross-crop masking",      "Fixes crop routing on diseased leaves"),
]
print(f"  {'Improvement':<40} {'Result':>35}")
for label, result in p3_data:
    print(f"  {label:<40} {result:>35}")
print()

print("KNOWN LIMITATIONS (Swin-Tiny)")
print("-" * 80)
print("  1. Heavily diseased leaves confuse crop classifier (okra PM -> broccoli)")
print("  2. tomato_bacterial_spot: F1=0.311 on PlantDoc (model conf 0.212)")
print("  3. Cropping any image hurts performance (global context dependency)")
print("  4. Severity head trained on placeholder data (predicts moderate)")
print("  5. No wild-condition eval exists for okra/brassica/chilli diseases")
print("  6. MoE cascading failure: wrong crop -> wrong diseases (mitigated")
print("     by soft masking but not eliminated)")
print()

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()
print(f"  {'Metric':<43} {'EffNetV2-S':>12} {'Swin-Tiny':>12}")
print("  " + "-" * 70)
print(f"  {'Controlled test F1':.<43} {'0.9523':>12} {'0.9410':>12}")
print(f"  {'PlantDoc wild F1 (7 tomato)':.<43} {'N/A':>12} {'0.5630':>12}")
print(f"  {'Chilli real-world accuracy':.<43} {'0% BROKEN':>12} {'90% FIXED':>12}")
print(f"  {'Inference time':.<43} {'~5-10s':>12} {'~1.7s':>12}")
print(f"  {'Okra PM real-world':.<43} {'Works':>12} {'Struggles':>12}")
print()
print("  EfficientNetV2-S: Higher controlled F1, but chilli broken,")
print("    no domain adaptation, slower inference")
print()
print("  Swin-Tiny: Slightly lower controlled F1 (-0.011), but chilli fixed,")
print("    CORAL domain adaptation, 3x faster inference, attention pooling")
print("    critical, but MoE routing can fail on heavily diseased leaves")
print("=" * 80)
