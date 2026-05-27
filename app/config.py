# app/config.py
# Single source of truth for ALL constants. Every other module imports from here.
# No magic numbers anywhere else in the codebase.

import os
import torch

# ── CLASS DEFINITIONS ──────────────────────────────────────────────────────
# Existing 10 classes (indices 0-9) MUST remain at their current positions.
# New classes (tomato 10-18, chilli 19-22) are appended after.
CLASS_NAMES = [
    # Okra (indices 0-4) — unchanged
    'okra_yvmv','okra_powdery_mildew','okra_cercospora','okra_enation',
    'okra_healthy',
    # Brassica (indices 5-9) — unchanged
    'brassica_black_rot','brassica_downy_mildew',
    'brassica_alternaria','brassica_clubroot','brassica_healthy',
    # Tomato (indices 10-18) — NEW
    'tomato_bacterial_spot','tomato_early_blight','tomato_late_blight',
    'tomato_leaf_mold','tomato_septoria_leaf_spot','tomato_target_spot',
    'tomato_mosaic_virus','tomato_yellow_leaf_curl_virus','tomato_healthy',
    # Chilli (indices 19-22) — NEW
    'chilli_anthracnose','chilli_cercospora_leaf_spot','chilli_leaf_curl',
    'chilli_healthy',
]
NUM_CLASSES      = len(CLASS_NAMES)  # 23
CLASS_TO_IDX     = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS     = {i: n for i, n in enumerate(CLASS_NAMES)}
OKRA_INDICES     = [0, 1, 2, 3, 4]
BRASSICA_INDICES = [5, 6, 7, 8, 9]
TOMATO_INDICES   = [10, 11, 12, 13, 14, 15, 16, 17, 18]
CHILLI_INDICES   = [19, 20, 21, 22]
HEALTHY_INDICES  = [4, 9, 18, 22]
NUM_CROPS        = 4
CROP_NAMES       = ["okra", "brassica", "tomato", "chilli"]  # index = crop label
CROP_LABEL_MAP   = {name: i for i, name in enumerate(CROP_NAMES)}
CROP_FROM_IDX    = {
    0:0, 1:0, 2:0, 3:0, 4:0,       # okra = crop 0
    5:1, 6:1, 7:1, 8:1, 9:1,       # brassica = crop 1
    10:2, 11:2, 12:2, 13:2, 14:2,  # tomato = crop 2
    15:2, 16:2, 17:2, 18:2,
    19:3, 20:3, 21:3, 22:3,        # chilli = crop 3
}
# Maps each crop index to the list of valid disease class indices for that crop.
# Used in inference for cross-crop masking: zeroes out disease logits for wrong crop.
CROP_TO_DISEASE_INDICES = {
    0: [0, 1, 2, 3, 4],                          # okra: indices 0-4
    1: [5, 6, 7, 8, 9],                          # brassica: indices 5-9
    2: [10, 11, 12, 13, 14, 15, 16, 17, 18],     # tomato: indices 10-18
    3: [19, 20, 21, 22],                          # chilli: indices 19-22
}
HEALTHY_CLASSES  = {'okra_healthy', 'brassica_healthy', 'tomato_healthy', 'chilli_healthy'}

# ── PLANTDOC CLASS MAP ──────────────────────────────────────────────────────
# [FIX GAP 52] Exact PlantDoc folder name -> canonical class mapping.
# Used by download_plantdoc.py and 08_evaluate_tier2_plantdoc.py.
PLANTDOC_CLASS_MAP = {
    'Cabbage__Black_Rot'              : 'brassica_black_rot',
    'Cabbage__Downy_Mildew'           : 'brassica_downy_mildew',
    'Cabbage__Alternaria_leaf_spot'   : 'brassica_alternaria',
    'Cabbage__healthy'                : 'brassica_healthy',
    'cabbage black rot'               : 'brassica_black_rot',
    'cabbage downy mildew'            : 'brassica_downy_mildew',
    'cabbage alternaria leaf spot'    : 'brassica_alternaria',
    'cabbage healthy'                 : 'brassica_healthy',
    # Tomato PlantDoc mappings
    'Tomato Early blight leaf'        : 'tomato_early_blight',
    'Tomato leaf late blight'         : 'tomato_late_blight',
    'Tomato leaf mosaic virus'        : 'tomato_mosaic_virus',
    'Tomato leaf yellow virus'        : 'tomato_yellow_leaf_curl_virus',
    'Tomato mold leaf'                : 'tomato_leaf_mold',
    'Tomato Septoria leaf spot'       : 'tomato_septoria_leaf_spot',
    'Tomato leaf bacterial spot'      : 'tomato_bacterial_spot',
    'Tomato two spotted spider mites leaf': None,  # discard, insect not disease
    'Tomato leaf'                     : None,      # discard, no disease label
}

# ── LABEL HARMONISATION MAPS ───────────────────────────────────────────────
# [FIX GAP 22] Defined here (not in 01_prepare_data.py) so both training
# scripts and agent scripts can import from app.config without circular imports.
LABEL_MAP = {
    # OKRA YVMV
    'okra_yellow_vein':'okra_yvmv','yvmv':'okra_yvmv',
    'yellow vein mosaic':'okra_yvmv','yellow_vein_mosaic':'okra_yvmv',
    'yellow_vein_mosaic_virus':'okra_yvmv','bhindi_mosaic':'okra_yvmv',
    'yellow vein mosaic virus':'okra_yvmv','okra_yvmv':'okra_yvmv',
    'yellowveinmosaic':'okra_yvmv','yellow vein':'okra_yvmv',
    'mosaic virus':'okra_yvmv','yvm':'okra_yvmv',
    'yellow_mosaic':'okra_yvmv','yellowmosaic':'okra_yvmv',
    # OKRA POWDERY MILDEW
    'okra_powdery_mildew':'okra_powdery_mildew',
    'powdery_mildew_okra':'okra_powdery_mildew',
    'powdery mildew okra':'okra_powdery_mildew',
    # OKRA CERCOSPORA
    'okra_leaf_spot':'okra_cercospora','cercospora':'okra_cercospora',
    'cercospora_leaf_spot':'okra_cercospora','okra_cercospora':'okra_cercospora',
    'cercospora_abelmoschi':'okra_cercospora','leaf spot okra':'okra_cercospora',
    # OKRA ENATION
    'enation_leaf_curl':'okra_enation','okra_leaf_curl':'okra_enation',
    'enation leaf curl':'okra_enation','okra_enation':'okra_enation',
    'leaf_curl_okra':'okra_enation','okra leaf curl':'okra_enation',
    'enation':'okra_enation',
    # OKRA HEALTHY
    'okra_healthy':'okra_healthy','healthy_okra':'okra_healthy',
    'okra healthy':'okra_healthy','okra_normal':'okra_healthy',
    'healthy okra':'okra_healthy',
    # BRASSICA BLACK ROT
    'black_rot':'brassica_black_rot','brassica_black_rot':'brassica_black_rot',
    'blackrot':'brassica_black_rot','black rot':'brassica_black_rot',
    'cabbage_black_rot':'brassica_black_rot','xanthomonas':'brassica_black_rot',
    'bacterial_black_rot':'brassica_black_rot',
    # BRASSICA DOWNY MILDEW
    'downy_mildew_brassica':'brassica_downy_mildew',
    'brassica_downy_mildew':'brassica_downy_mildew',
    'cabbage_downy_mildew':'brassica_downy_mildew',
    'downy mildew brassica':'brassica_downy_mildew',
    'hyaloperonospora':'brassica_downy_mildew',
    'downy mildew cabbage':'brassica_downy_mildew',
    # BRASSICA ALTERNARIA
    'alternaria_brassicae':'brassica_alternaria',
    'alternaria_leaf_spot_brassica':'brassica_alternaria',
    'brassica_alternaria':'brassica_alternaria',
    'cabbage_alternaria':'brassica_alternaria',
    'dark_leaf_spot':'brassica_alternaria',
    'alternaria leaf spot':'brassica_alternaria',
    'alternaria brassica':'brassica_alternaria',
    # BRASSICA CLUBROOT
    'clubroot':'brassica_clubroot','brassica_clubroot':'brassica_clubroot',
    'club root':'brassica_clubroot','club_root':'brassica_clubroot',
    'plasmodiophora':'brassica_clubroot',
    # BRASSICA HEALTHY
    'brassica_healthy':'brassica_healthy','cabbage_healthy':'brassica_healthy',
    'cauliflower_healthy':'brassica_healthy','healthy_brassica':'brassica_healthy',
    'healthy cabbage':'brassica_healthy','healthy_cabbage':'brassica_healthy',
    'broccoli_healthy':'brassica_healthy','healthy_broccoli':'brassica_healthy',
    'healthy brassica':'brassica_healthy',
}

# ── SOURCE-SPECIFIC LABEL OVERRIDES (nested dict: source -> folder -> class) ──
# Each source dataset maps its EXACT folder names to canonical class names.
# Any folder NOT listed here for a given source is silently skipped.
SOURCE_LABEL_OVERRIDES = {
    # manojgadde/yellow-vein-mosaic-disease
    'gadde_okra': {
        'diseased okra leaf': 'okra_yvmv',
        'fresh okra leaf':    'okra_healthy',
    },
    # asrafulme/dataset-diya — Broccoli subfolder
    'diya_broccoli': {
        'Alternaria black spot of brocoli': 'brassica_alternaria',
        'Black rot of brocoli':             'brassica_black_rot',
        'Health leaf of brocoli':           'brassica_healthy',
    },
    # asrafulme/dataset-diya — Cabbage subfolder
    'diya_cabbage': {
        'Alternaria leaf spot of cabbage':  'brassica_alternaria',
        'Healthy leaf of cabbage':          'brassica_healthy',
    },
    # asrafulme/dataset-diya — Cauliflower subfolder
    'diya_cauliflower': {
        'Alternaria leaf spot of cauliflower': 'brassica_alternaria',
        'Healthy leaf of cauliflower':          'brassica_healthy',
    },
    # noamaanabdulazeem/cauliflower-dataset
    'cauliflower_noam': {
        'Black Rot':    'brassica_black_rot',
        'Downy Mildew': 'brassica_downy_mildew',
        'No disease':   'brassica_healthy',
    },
    # mubbassir/balanced-cabbage-dataset-200-each
    'cabbage_balanced': {
        'Alternaria_Leaf_Spot': 'brassica_alternaria',
        'Black Rot':            'brassica_black_rot',
        'Downy Mildew':         'brassica_downy_mildew',
        'No disease':           'brassica_healthy',
        'club root':            'brassica_clubroot',
    },

    # ── NEW DATASETS (added after Step 03) ─────────────────────────────────

    # manhhoangvan/yeesidtaset — okra powdery mildew + healthy
    # NOTE: dataset has typo 'okra_powderly' (with 'l'), not 'okra_powdery'
    'yeesi': {
        'okra_powderly': 'okra_powdery_mildew',
        'okra_healthy':  'okra_healthy',
    },

    # Mendeley: Okra DiseaseNet (nh7zk4hv8z) — Training split only
    'mendeley_okra': {
        'Class 1 - Cercospora Leaf Spot': 'okra_cercospora',
        'Class 1 - Cercospora leaf spot': 'okra_cercospora',
        'Class 4 - Leaf curly virus':     'okra_enation',
        'Class 4 \u2013 Leaf curly virus':     'okra_enation',
        'Class 3 - Healthy':              'okra_healthy',
        'Class 3- Healthy':               'okra_healthy',
    },

    # Mendeley: Cauliflower Leaf Diseases (x995snz7p3)
    'mendeley_caul_leaf': {
        'Black Rot': 'brassica_black_rot',
        'Healthy':   'brassica_healthy',
    },

    # Mendeley: Cabbage Crop Diseases (sjmgzhwrxv) — originals only
    # Actual folder names from ORIGINAL_ALL_COMBINE_CABBAGE_DISEASES.zip:
    'mendeley_cabbage_dis': {
        'Cabbage_Altenaria_spot':           'brassica_alternaria',
        'Cabbage - Black_rot':              'brassica_black_rot',
        'Cabbage downy mildew - Original':  'brassica_downy_mildew',
        'Cabbage - Healthy':                'brassica_healthy',
    },

    # sharifashik/cauliflower-image-dataset — has DOUBLE trailing underscores
    'caul_sharifashik': {
        'AlternariaLeafSpot__': 'brassica_alternaria',
        'BlackRot__':           'brassica_black_rot',
        'DowneyMildew__':       'brassica_downy_mildew',
        'ClubRoot__':           'brassica_clubroot',
    },

    # VegNet cauliflower (local zip: Original Dataset.zip)
    'mendeley_vegnet': {
        'Downy Mildew': 'brassica_downy_mildew',
        'Black Rot':    'brassica_black_rot',
        'No disease':   'brassica_healthy',
    },

    # ── BANGLADESH OKRA (Mendeley ck7vkp23c7) ─────────────────────────────
    'bangladesh_okra': {
        'Leaf_curl':                'okra_enation',
        'Leaf curl':                'okra_enation',
        'Leaf Curl':                'okra_enation',
        'Leaf_spot':                'okra_cercospora',
        'Leaf spot':                'okra_cercospora',
        'Leaf Spot':                'okra_cercospora',
        'Fresh':                    'okra_healthy',
        'Healthy':                  'okra_healthy',
        'healthy':                  'okra_healthy',
        'Yellow_vein_mosaic_virus': 'okra_yvmv',
        'Yellow vein mosaic':       'okra_yvmv',
        'YVMV':                     'okra_yvmv',
        # discard: Insects_eaten_leaves, Predisposition
    },

    # ── TOMATO DATASETS ────────────────────────────────────────────────────

    # PlantVillage tomato (Mendeley tywbtsjrjv or Kaggle abdallahalidev)
    'plantvillage_tomato': {
        'Tomato___Bacterial_spot':                      'tomato_bacterial_spot',
        'Tomato___Early_blight':                        'tomato_early_blight',
        'Tomato___Late_blight':                         'tomato_late_blight',
        'Tomato___Leaf_Mold':                           'tomato_leaf_mold',
        'Tomato___Septoria_leaf_spot':                  'tomato_septoria_leaf_spot',
        'Tomato___Target_Spot':                         'tomato_target_spot',
        'Tomato___Tomato_mosaic_virus':                 'tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus':       'tomato_yellow_leaf_curl_virus',
        'Tomato___healthy':                             'tomato_healthy',
        # Alternate naming (some versions use double underscore)
        'Tomato__Bacterial_spot':                       'tomato_bacterial_spot',
        'Tomato__Early_blight':                         'tomato_early_blight',
        'Tomato__Late_blight':                          'tomato_late_blight',
        'Tomato__Leaf_Mold':                            'tomato_leaf_mold',
        'Tomato__Septoria_leaf_spot':                   'tomato_septoria_leaf_spot',
        'Tomato__Target_Spot':                          'tomato_target_spot',
        'Tomato__Tomato_mosaic_virus':                  'tomato_mosaic_virus',
        'Tomato__Tomato_Yellow_Leaf_Curl_Virus':        'tomato_yellow_leaf_curl_virus',
        'Tomato__healthy':                              'tomato_healthy',
        # Bare names (no Tomato__ prefix)
        'Bacterial_spot':                               'tomato_bacterial_spot',
        'Early_blight':                                 'tomato_early_blight',
        'Late_blight':                                  'tomato_late_blight',
        'Leaf_Mold':                                    'tomato_leaf_mold',
        'Septoria_leaf_spot':                            'tomato_septoria_leaf_spot',
        'Target_Spot':                                  'tomato_target_spot',
        'Tomato_mosaic_virus':                          'tomato_mosaic_virus',
        'Tomato_Yellow_Leaf_Curl_Virus':                'tomato_yellow_leaf_curl_virus',
        'healthy':                                      'tomato_healthy',
        # discard: Spider_mites, all non-tomato
    },

    # ashishmotwani/tomato (Kaggle) — real-world + lab images
    'tomato_ashish': {
        'Tomato___Bacterial_spot':                      'tomato_bacterial_spot',
        'Tomato___Early_blight':                        'tomato_early_blight',
        'Tomato___Late_blight':                         'tomato_late_blight',
        'Tomato___Leaf_Mold':                           'tomato_leaf_mold',
        'Tomato___Septoria_leaf_spot':                  'tomato_septoria_leaf_spot',
        'Tomato___Target_Spot':                         'tomato_target_spot',
        'Tomato___Tomato_mosaic_virus':                 'tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus':       'tomato_yellow_leaf_curl_virus',
        'Tomato___healthy':                             'tomato_healthy',
        'Bacterial_spot':                               'tomato_bacterial_spot',
        'Early_blight':                                 'tomato_early_blight',
        'Late_blight':                                  'tomato_late_blight',
        'Leaf_Mold':                                    'tomato_leaf_mold',
        'Septoria_leaf_spot':                            'tomato_septoria_leaf_spot',
        'Target_Spot':                                  'tomato_target_spot',
        'Tomato_mosaic_virus':                          'tomato_mosaic_virus',
        'Yellow_Leaf_Curl_Virus':                       'tomato_yellow_leaf_curl_virus',
        'Tomato_Yellow_Leaf_Curl_Virus':                'tomato_yellow_leaf_curl_virus',
        'healthy':                                      'tomato_healthy',
        # discard: Spider_mites, Powdery_Mildew
    },

    # cookiefinder/tomato-disease-multiple-sources (Kaggle)
    'tomato_cookiefinder': {
        'Tomato___Bacterial_spot':                      'tomato_bacterial_spot',
        'Tomato___Early_blight':                        'tomato_early_blight',
        'Tomato___Late_blight':                         'tomato_late_blight',
        'Tomato___Leaf_Mold':                           'tomato_leaf_mold',
        'Tomato___Septoria_leaf_spot':                  'tomato_septoria_leaf_spot',
        'Tomato___Target_Spot':                         'tomato_target_spot',
        'Tomato___Tomato_mosaic_virus':                 'tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus':       'tomato_yellow_leaf_curl_virus',
        'Tomato___healthy':                             'tomato_healthy',
        'Bacterial_spot':                               'tomato_bacterial_spot',
        'Early_blight':                                 'tomato_early_blight',
        'Late_blight':                                  'tomato_late_blight',
        'Leaf_Mold':                                    'tomato_leaf_mold',
        'Septoria_leaf_spot':                            'tomato_septoria_leaf_spot',
        'Target_Spot':                                  'tomato_target_spot',
        'Tomato_mosaic_virus':                          'tomato_mosaic_virus',
        'Tomato_Yellow_Leaf_Curl_Virus':                'tomato_yellow_leaf_curl_virus',
        'healthy':                                      'tomato_healthy',
        # discard: Spider_mites, Powdery_Mildew
    },

    # luisolazo/tomato-diseases (Kaggle) — lowercase folder names
    'tomato_luisolazo': {
        'bacterial_spot':           'tomato_bacterial_spot',
        'early_blight':             'tomato_early_blight',
        'late_blight':              'tomato_late_blight',
        'leaf_mold':                'tomato_leaf_mold',
        'septoria_leaf_spot':       'tomato_septoria_leaf_spot',
        'target_spot':              'tomato_target_spot',
        'mosaic_virus':             'tomato_mosaic_virus',
        'yellow_leaf_curl_virus':   'tomato_yellow_leaf_curl_virus',
        'healthy':                  'tomato_healthy',
        # discard: twospotted_spider_mite
    },

    # hakim11/tomato-disease (Kaggle) — unique folder naming
    'tomato_hakim': {
        'Tomato_Early_blight':                          'tomato_early_blight',
        'Tomato_Late_blight':                           'tomato_late_blight',
        'Tomato_Leaf_Mold':                             'tomato_leaf_mold',
        'Tomato_Septoria_leaf_spot':                    'tomato_septoria_leaf_spot',
        'Tomato__Target_Spot':                          'tomato_target_spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus':        'tomato_yellow_leaf_curl_virus',
        'Tomato__Tomato_mosaic_virus':                  'tomato_mosaic_virus',
        'Tomato_healthy':                               'tomato_healthy',
        # discard: Tomato_Spider_mites_Two_spotted_spider_mite
        # note: no Bacterial_spot in this dataset
    },

    # Mendeley tomato leaf disease (zfv4jj7855)
    'tomato_mendeley': {
        'Tomato___Bacterial_spot':                      'tomato_bacterial_spot',
        'Tomato___Early_blight':                        'tomato_early_blight',
        'Tomato___Late_blight':                         'tomato_late_blight',
        'Tomato___Leaf_Mold':                           'tomato_leaf_mold',
        'Tomato___Septoria_leaf_spot':                  'tomato_septoria_leaf_spot',
        'Tomato___Target_Spot':                         'tomato_target_spot',
        'Tomato___Tomato_mosaic_virus':                 'tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus':       'tomato_yellow_leaf_curl_virus',
        'Tomato___healthy':                             'tomato_healthy',
        'Bacterial_spot':                               'tomato_bacterial_spot',
        'Early_blight':                                 'tomato_early_blight',
        'Late_blight':                                  'tomato_late_blight',
        'Leaf_Mold':                                    'tomato_leaf_mold',
        'Septoria_leaf_spot':                            'tomato_septoria_leaf_spot',
        'Target_Spot':                                  'tomato_target_spot',
        'Tomato_mosaic_virus':                          'tomato_mosaic_virus',
        'Tomato_Yellow_Leaf_Curl_Virus':                'tomato_yellow_leaf_curl_virus',
        'healthy':                                      'tomato_healthy',
    },

    # kaustubhb999/tomatoleaf (Kaggle) — balanced PlantVillage subset
    'tomato_kaustubh': {
        'Tomato___Bacterial_spot':                      'tomato_bacterial_spot',
        'Tomato___Early_blight':                        'tomato_early_blight',
        'Tomato___Late_blight':                         'tomato_late_blight',
        'Tomato___Leaf_Mold':                           'tomato_leaf_mold',
        'Tomato___Septoria_leaf_spot':                  'tomato_septoria_leaf_spot',
        'Tomato___Target_Spot':                         'tomato_target_spot',
        'Tomato___Tomato_mosaic_virus':                 'tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus':       'tomato_yellow_leaf_curl_virus',
        'Tomato___healthy':                             'tomato_healthy',
        'Bacterial_spot':                               'tomato_bacterial_spot',
        'Early_blight':                                 'tomato_early_blight',
        'Late_blight':                                  'tomato_late_blight',
        'Leaf_Mold':                                    'tomato_leaf_mold',
        'Septoria_leaf_spot':                            'tomato_septoria_leaf_spot',
        'Target_Spot':                                  'tomato_target_spot',
        'Tomato_mosaic_virus':                          'tomato_mosaic_virus',
        'Tomato_Yellow_Leaf_Curl_Virus':                'tomato_yellow_leaf_curl_virus',
        'healthy':                                      'tomato_healthy',
    },

    # ── CHILLI DATASET ─────────────────────────────────────────────────────

    # Mendeley: Chilli Leaf Disease Bangladesh (wzc6r6w5w5)
    'chilli_bangladesh': {
        'Anthracnose':            'chilli_anthracnose',
        'anthracnose':            'chilli_anthracnose',
        'Cercospora Leaf Spot':   'chilli_cercospora_leaf_spot',
        'Cercospora leaf spot':   'chilli_cercospora_leaf_spot',
        'cercospora leaf spot':   'chilli_cercospora_leaf_spot',
        'Leaf Curl Disease':      'chilli_leaf_curl',
        'Leaf Curl':              'chilli_leaf_curl',
        'Leaf curl':              'chilli_leaf_curl',
        'leaf curl':              'chilli_leaf_curl',
        'Healthy Leaves':         'chilli_healthy',
        'Fresh Leaf':             'chilli_healthy',
        'Healthy':                'chilli_healthy',
        'healthy':                'chilli_healthy',
    },

    # ── NEW CHILLI DATASETS (round 2) ─────────────────────────────────────

    # Prudhvi: Anthracnose in Chilli Mobile Captured (Kaggle)
    'chilli_anthracnose_prudhvi': {
        'Anthracnose':  'chilli_anthracnose',
        'anthracnose':  'chilli_anthracnose',
        'Healthy':      'chilli_healthy',
        'healthy':      'chilli_healthy',
    },

    # Karnataka COLD Chilli Dataset (resized_raw images)
    # Note: 'cerocospora' is a typo in the dataset (missing an 'r')
    'chilli_cold_karnataka': {
        'cerocospora':  'chilli_cercospora_leaf_spot',
        'cercospora':   'chilli_cercospora_leaf_spot',
        'Cerocospora':  'chilli_cercospora_leaf_spot',
        'Cercospora':   'chilli_cercospora_leaf_spot',
        'healthy':      'chilli_healthy',
        'Healthy':      'chilli_healthy',
        # discard: murda complex, nutritional deficiency, powdery mildew
    },

    # Bangladesh 2025 Chilli Leaf Disease Original (Mendeley)
    'chilli_bangladesh_2025': {
        'Curl Virus':             'chilli_leaf_curl',
        'curl virus':             'chilli_leaf_curl',
        'Curl_Virus':             'chilli_leaf_curl',
        'CurlVirus':              'chilli_leaf_curl',
        'Cercospora Leaf Spot':   'chilli_cercospora_leaf_spot',
        'cercospora leaf spot':   'chilli_cercospora_leaf_spot',
        'Cercospora_Leaf_Spot':   'chilli_cercospora_leaf_spot',
        'Healthy Leaves':         'chilli_healthy',
        'Healthy Leaf':           'chilli_healthy',
        'healthy leaves':         'chilli_healthy',
        'Healthy':                'chilli_healthy',
        'healthy':                'chilli_healthy',
        # discard: Bacterial Spot, Nutrition Deficiency, White spot
    },

    # Annotated Smartphone Chilli Images (Mendeley wzc6r6w5w5/2)
    'chilli_annotated_smartphone': {
        'Anthracnose':            'chilli_anthracnose',
        'anthracnose':            'chilli_anthracnose',
        'Cercospora Leaf Spot':   'chilli_cercospora_leaf_spot',
        'cercospora leaf spot':   'chilli_cercospora_leaf_spot',
        'Cercospora_Leaf_Spot':   'chilli_cercospora_leaf_spot',
        'Leaf Curl Disease':      'chilli_leaf_curl',
        'leaf curl disease':      'chilli_leaf_curl',
        'Leaf_Curl_Disease':      'chilli_leaf_curl',
        'Healthy Leaves':         'chilli_healthy',
        'Fresh Leaf':             'chilli_healthy',
        'healthy leaves':         'chilli_healthy',
        'Healthy':                'chilli_healthy',
        'healthy':                'chilli_healthy',
    },
}

# ── MODEL ARCHITECTURE (Phase 1: Swin-Tiny) ──────────────────────────────
BACKBONE_NAME   = 'swin_tiny_patch4_window7_224'
# timm Swin-Tiny: features_only=True, out_indices=(1,2,3)
# Stage 1: 192ch 28x28 (NHWC → permute to NCHW) = P3
# Stage 2: 384ch 14x14 = P4
# Stage 3: 768ch  7x7  = P5
# CRITICAL: Swin outputs NHWC — must permute(0,3,1,2) before FPN Conv2d
FPN_IN_CH       = [192, 384, 768]  # from Swin-Tiny stages 1,2,3 (verified in Group 1.1)
FPN_OUT_CH      = 256              # all FPN levels projected to this (unchanged)
POOLED_DIM      = 256              # after attention pooling on FPN output
CROP_EMB_DIM    = 64               # crop classifier embedding — UNCHANGED for cache compat
HEAD_HIDDEN_DIM = 256              # hidden layer in severity head
DROPOUT_P       = 0.3

# Attention Pooling (replaces Global Average Pooling)
ATT_POOL_SPATIAL = 7               # FPN P3 spatial at 28x28, but attention uses flattened
ATT_POOL_SIZE    = 49              # 7 * 7

# Conditional Layer Norm (replaces FiLM conditioning)
CLN_FEATURE_DIM  = 256             # same as FPN_OUT_CH
CLN_NUM_CROPS    = 4               # one gamma+beta set per crop per channel

# Mixture of Experts disease head (replaces unified 23-class head)
MOE_HIDDEN_DIM   = 128             # hidden dim in each expert MLP
MOE_NUM_EXPERTS  = 4               # one expert per crop
MOE_DROPOUT      = 0.1             # dropout within each expert

# DeiT distillation parameters (used in Phase 2 training only)
DISTILLATION_ALPHA = 0.5           # hard label weight (1-alpha = soft label weight)
DISTILLATION_TEMP  = 3.0           # temperature for teacher soft targets
IMG_H = IMG_W   = 224
IMG_SIZE        = (224, 224)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ── TRAINING ───────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
PHASE1_EPOCHS    = 10
PHASE2_EPOCHS    = 7
PHASE1_LR        = 1e-3
PHASE2_BASE_LR   = 1e-4
LLRD_DECAY       = 0.85
GRAD_CLIP_NORM   = 1.0
BATCH_SIZE       = 32     # use 16 + GRAD_ACCUM_STEPS=2 if VRAM OOM
GRAD_ACCUM_STEPS = 1
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTH     = 0.1
LOSS_W_CROP      = 0.4
LOSS_W_DISEASE   = 0.4
LOSS_W_SEVERITY  = 0.2
MAX_POS_WEIGHT   = 10.0   # cap to prevent loss destabilisation
EARLY_STOP_PAT   = 5
EARLY_STOP_DELTA = 0.001
KEEP_CKPTS       = 3
# [FIX GAP 35] OneCycleLR constants — used in 05_train_phase2.py.
# Import these; do NOT hardcode 0.1, 10, 1000 in training scripts.
ONE_CYCLE_PCT    = 0.15   # pct_start — warmup fraction (matches PHASE2_WARMUP_FRACTION)
ONE_CYCLE_DIV    = 10     # div_factor — initial LR = max_lr / div_factor
ONE_CYCLE_FDIV   = 1000   # final_div_factor — final LR = max_lr / final_div

# ── SEVERITY PROXY GENERATION ──────────────────────────────────────────────
SEVERITY_PROXY_THRESHOLD = 0.30   # top 30% activations = lesion region
SEVERITY_MILD_MAX        = 0.15   # coverage < 0.15 = mild
SEVERITY_MOD_MAX         = 0.50   # coverage 0.15-0.50 = moderate, else severe

# ── DATA PIPELINE ──────────────────────────────────────────────────────────
# [FIX GAP 60] HEIC removed from VALID_EXT — pillow-heif not installed.
VALID_EXT        = {'.jpg', '.jpeg', '.png', '.webp',
                    '.JPG', '.JPEG', '.PNG', '.WEBP'}
SPLIT_TRAIN      = 0.70
SPLIT_VAL        = 0.15
SPLIT_TEST       = 0.15
MIN_IMGS_CLASS   = 150
CLUBROOT_OVERSAMPLE = 2.0

# ── INPUT VALIDATION ───────────────────────────────────────────────────────
MAX_FILE_MB      = 10
MIN_BLUR_VAR     = 80
MIN_PIXEL_MEAN   = 40
MAX_PIXEL_MEAN   = 220
MIN_IMG_DIM      = 150
MAX_CH_RATIO     = 0.65   # no single channel > 65% of total (non-plant check)

# ── INFERENCE ──────────────────────────────────────────────────────────────
DISEASE_THRESH   = 0.50   # global fallback for evaluation scripts (uniform F1 computation)
MC_PASSES        = 5
TEMP_INIT        = 1.5    # LBFGS starting value for temperature scaling

# OOD gate — lowered from 0.75 to 0.50 to reduce false OOD on chilli/brassica
OOD_CROP_CONFIDENCE_THRESHOLD = 0.25  # Lowered from 0.50 — heavily diseased leaves confuse crop classifier

# Per-class disease thresholds — set to 0.50 as balanced middle ground
# Lab-optimized (0.56-0.88) missed real photos; ultra-low (0.35-0.40) caused
# too many co-predictions. 0.50 is the balanced default.
DISEASE_THRESHOLDS = {
    'okra_yvmv':                     0.30,   # Lowered for real-world photos
    'okra_powdery_mildew':           0.30,   # Lowered for real-world photos
    'okra_cercospora':               0.30,   # Lowered for real-world photos
    'okra_enation':                  0.30,   # Lowered for real-world photos
    'okra_healthy':                  0.35,   # Slightly higher for healthy class
    'brassica_black_rot':            0.30,   # Lowered for real-world photos
    'brassica_downy_mildew':         0.30,   # Lowered for real-world photos
    'brassica_alternaria':           0.30,   # Lowered for real-world photos
    'brassica_clubroot':             0.30,   # Lowered for real-world photos
    'brassica_healthy':              0.35,   # Slightly higher for healthy class
    'tomato_bacterial_spot':         0.175,  # PlantDoc-tuned (was 0.50, +0.215 F1)
    'tomato_early_blight':           0.600,  # PlantDoc-tuned (was 0.50, +0.033 F1)
    'tomato_late_blight':            0.300,  # PlantDoc-tuned (was 0.50, +0.107 F1)
    'tomato_leaf_mold':              0.325,  # PlantDoc-tuned (was 0.50, +0.057 F1)
    'tomato_septoria_leaf_spot':     0.300,  # PlantDoc-tuned (was 0.50, +0.233 F1)
    'tomato_target_spot':            0.50,
    'tomato_mosaic_virus':           0.400,  # PlantDoc-tuned (was 0.50, +0.135 F1)
    'tomato_yellow_leaf_curl_virus': 0.375,  # PlantDoc-tuned (was 0.50, +0.072 F1)
    'tomato_healthy':                0.50,
    'chilli_anthracnose':            0.30,   # Lowered for real-world photos
    'chilli_cercospora_leaf_spot':   0.30,   # Lowered for real-world photos
    'chilli_leaf_curl':              0.30,   # Lowered for real-world photos
    'chilli_healthy':                0.35,   # Slightly higher for healthy class
}

# Post-processing constants
COPRED_GAP_THRESH    = 0.05   # suppress weaker co-prediction if gap > this (very tight — only near-identical confidence passes)
MAX_COPREDICTIONS    = 1      # default to 1 disease per image (co-infection is rare in practice)

# Healthy suppression — if a healthy class exceeds this confidence,
# suppress disease predictions below HEALTHY_SUPPRESSION_DISEASE_MIN
HEALTHY_SUPPRESSION_CONFIDENCE  = 0.55
HEALTHY_SUPPRESSION_DISEASE_MIN = 0.50

# CLAHE clip limit — increased from 2.0 to 4.0 for better contrast on real photos
CLAHE_CLIP_LIMIT = 4.0

# Test-Time Augmentation — horizontal flip ensemble
TTA_ENABLED = True

# Adaptive threshold: lower disease thresholds when crop confidence is moderate
ADAPTIVE_THRESH_CROP_HIGH   = 0.85   # above this, use normal thresholds
ADAPTIVE_THRESH_REDUCTION   = 0.10   # below CROP_HIGH, reduce thresholds by this

# Disease co-occurrence priors (True = plausible co-infection, False = unlikely)
DISEASE_COOCCURRENCE = {
    ('tomato_early_blight', 'tomato_septoria_leaf_spot'): True,
    ('tomato_early_blight', 'tomato_target_spot'): True,
    ('tomato_bacterial_spot', 'tomato_septoria_leaf_spot'): True,
    ('tomato_late_blight', 'tomato_early_blight'): False,   # rarely co-occur
    ('tomato_mosaic_virus', 'tomato_bacterial_spot'): False, # virus + bacteria unlikely
    ('tomato_mosaic_virus', 'tomato_leaf_mold'): False,
    ('chilli_leaf_curl', 'chilli_anthracnose'): False,      # virus + fungus unlikely
}

# Weighted sampler cap — prevents extreme oversampling of thin classes
# With 41:1 imbalance, rarest class sampled at most 10x the most common
SAMPLER_MAX_WEIGHT_RATIO = 10.0

# Label smoothing for disease head BCE loss only
LABEL_SMOOTHING = 0.1

# Phase 2 warmup fraction (fraction of training steps used for warmup)
PHASE2_WARMUP_FRACTION = 0.15

# Early stopping patience for Phase 2 (epochs without improvement)
EARLY_STOPPING_PATIENCE = 3

# ── EVALUATION THRESHOLDS ──────────────────────────────────────────────────
TIER2_MIN_F1    = 0.55
TIER3_MIN_ACC   = 0.70
TIER3_MIN_IMGS  = 50
TIER3_MIN_CLS   = 5      # minimum images per class to evaluate that class

# ── FILE PATHS (all relative to project ROOT — [FIX GAP 30]) ───────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA        = os.path.join(ROOT, 'data')
RAW         = os.path.join(ROOT, 'data', 'raw')
PROC        = os.path.join(ROOT, 'data', 'processed')  # created but empty
TRAIN_DIR   = os.path.join(ROOT, 'data', 'processed', 'train')   # not used
VAL_DIR     = os.path.join(ROOT, 'data', 'processed', 'val')     # not used
KERALA_DIR  = os.path.join(ROOT, 'data', 'kerala')
PLANTDOC_DIR= os.path.join(ROOT, 'data', 'plantdoc')
META        = os.path.join(ROOT, 'data', 'metadata')
SOURCE_MAP  = os.path.join(ROOT, 'data', 'metadata', 'source_map.csv')
SEV_LABELS  = os.path.join(ROOT, 'data', 'metadata', 'severity_labels.csv')
# [FIX GAP 62] CLASS_COUNTS_PATH was missing from v5:
CLASS_COUNTS_PATH = os.path.join(ROOT, 'data', 'metadata', 'class_counts.csv')
MODELS      = os.path.join(ROOT, 'models')
CKPT_DIR    = os.path.join(ROOT, 'models', 'checkpoints')
# Teacher model: EfficientNetV2-S trained in previous session — NEVER OVERWRITE
TEACHER_MODEL = os.path.join(ROOT, 'models', 'best_model.pt')
TEACHER_BACKBONE_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'
TEACHER_NUM_CLASSES = 23
TEACHER_FPN_IN_CH = [64, 160, 256]

# Student model: Swin-Tiny — all training saves here, NOT to TEACHER_MODEL
BEST_MODEL  = os.path.join(ROOT, 'models', 'swin_best_model.pt')
TEMP_PATH   = os.path.join(ROOT, 'models', 'temperature.pt')
MOBILE_SAM_CHECKPOINT = os.path.join(ROOT, 'models', 'mobile_sam.pt')

# Feature cache dimensions
CACHE_POOLED_DIM   = 256    # after FPN + attention pooling
CACHE_CROP_EMB_DIM = 64     # intermediate crop embedding — UNCHANGED
CACHE       = os.path.join(ROOT, 'cache')
TRAIN_CACHE = os.path.join(ROOT, 'cache', 'train_features.pt')
VAL_CACHE   = os.path.join(ROOT, 'cache', 'val_features.pt')
REPORTS     = os.path.join(ROOT, 'reports')
DIAG_JSON   = os.path.join(ROOT, 'diagnosis', 'diagnosis_lookup.json')

# ── DEVICE ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── WANDB ──────────────────────────────────────────────────────────────────
WANDB_PROJECT = 'plant-disease-kerala'
WANDB_CONFIG  = {
    'backbone'         : BACKBONE_NAME,
    'img_size'         : IMG_SIZE,
    'batch_size'       : BATCH_SIZE,
    'phase1_epochs'    : PHASE1_EPOCHS,
    'phase2_epochs'    : PHASE2_EPOCHS,
    'phase1_lr'        : PHASE1_LR,
    'phase2_base_lr'   : PHASE2_BASE_LR,
    'llrd_decay'       : LLRD_DECAY,
    'dropout_p'        : DROPOUT_P,
    'loss_w_crop'      : LOSS_W_CROP,
    'loss_w_disease'   : LOSS_W_DISEASE,
    'loss_w_severity'  : LOSS_W_SEVERITY,
    'grad_clip_norm'   : GRAD_CLIP_NORM,
    'weight_decay'     : WEIGHT_DECAY,
    'label_smooth'     : LABEL_SMOOTH,
}

# ============================================================
# PHASE 0: DATA PREPARATION CONSTANTS
# Added during Phase 0 data preparation
# ============================================================

# Tomato data cap — T-SNE selected diverse subset
TOMATO_CAP_ENABLED = True
TOMATO_CAP_CSV = os.path.join(ROOT, 'data_prep', 'tomato_selected_8000.csv')

# iNaturalist source IDs — must always go to training split only
INATURALIST_SOURCES = [
    'inaturalist_okra',
    'inaturalist_chilli',
    'inaturalist_tomato',
    'inaturalist_brassica',
]
TRAINING_ONLY_SOURCES = INATURALIST_SOURCES

# PlantDoc splits
PLANTDOC_TRAIN_MANIFEST = os.path.join(ROOT, 'data_prep', 'plantdoc_train_200.csv')
PLANTDOC_EVAL_MANIFEST = os.path.join(ROOT, 'data_prep', 'plantdoc_eval_462.csv')

# Frozen test set — these image paths always go to test split
FROZEN_TEST_SET_CSV = os.path.join(ROOT, 'data_prep', 'frozen_test_set.csv')

# Validation quality audit — images flagged for removal
VAL_IMAGES_TO_REMOVE_TXT = os.path.join(ROOT, 'data_prep', 'val_images_to_remove.txt')

# iNaturalist pseudo-label manifest
INATURALIST_MANIFEST = os.path.join(ROOT, 'data_prep', 'inaturalist_manifest_pseudolabelled.csv')

# Two-tier weighted sampler constants
SAMPLER_USE_TWO_TIER = True
CROP_SAMPLING_WEIGHTS = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # Equal per crop

# ============================================================
# PHASE 2: TRAINING STRATEGY CONSTANTS
# ============================================================

# Focal Loss
FOCAL_GAMMA          = 2.0     # focusing parameter (0 = standard BCE)
FOCAL_GAMMA_WARMUP   = True    # ramp gamma 0→2 over first epoch

# CORAL Domain Adaptation
CORAL_LAMBDA              = 0.01       # weight of CORAL loss
PLANTDOC_SOURCE_PATTERN   = 'plantdoc' # str.contains() pattern for target domain
# NOTE: source_map.csv column is 'source_dataset', not 'source'
MIN_PLANTDOC_PER_BATCH    = 4          # DomainBalancedSampler guarantee

# ArcFace Auxiliary Loss
ARCFACE_WEIGHT       = 0.1
ARCFACE_SCALE        = 30.0
ARCFACE_MARGIN       = 0.3    # radians
ARCFACE_IN_FEATURES  = 256    # matches FPN_OUT_CH / POOLED_DIM

# EMA and SWA
EMA_DECAY            = 0.999
SWA_START_FRACTION   = 0.80   # start SWA after 80% of Phase 2B epochs
SWA_LR               = 1e-5

# Data Augmentation (Phase 2B only — not Phase 2A cached training)
CUTMIX_ALPHA         = 1.0
CUTMIX_PROB          = 0.5
RANDAUGMENT_N        = 2
RANDAUGMENT_M_DEFAULT = 10
RANDAUGMENT_M_THIN   = 20     # stronger for thin classes
RANDOM_ERASING_PROB  = 0.3

# Thin class indices — fewer than 500 training images after Phase 0
THIN_CLASS_INDICES = [
    1,   # okra_powdery_mildew
    2,   # okra_cercospora
    3,   # okra_enation
    6,   # brassica_downy_mildew
    8,   # brassica_clubroot
    15,  # tomato_target_spot
    19,  # chilli_anthracnose
]

# Phase 2A: Head training on cached features
PHASE2A_LR           = 1e-3
PHASE2A_EPOCHS       = 15
PHASE2A_BATCH        = 512

# Phase 2B: Full fine-tuning with all loss functions
PHASE2B_PEAK_LR      = 3e-4
PHASE2B_BACKBONE_LR  = 3e-5   # 1/10 of peak for backbone
PHASE2B_FPN_LR       = 1e-4   # 1/3 of peak for FPN+pool
PHASE2B_WARMUP_EPOCHS = 5
PHASE2B_MIN_LR       = 1e-6
PHASE2B_EPOCHS       = 30     # max — early stopping expected at 6-10
PHASE2B_BATCH        = 20     # reduce to 16 if OOM

# CosineAnnealingWarmRestarts (NOT OneCycleLR — incompatible with early stopping)
COSINE_T0            = 10     # restart period in epochs
COSINE_T_MULT        = 1      # period multiplier after each restart
COSINE_ETA_MIN       = 1e-6

# Distillation: do not apply to PlantDoc images (teacher biased on lab images)
DISTILLATION_SOURCE_ONLY = True

# Loss weights for Phase 2B total loss composition
LOSS_WEIGHT_CROP     = 0.3
LOSS_WEIGHT_SEVERITY = 0.1    # low: severity uses placeholder labels
