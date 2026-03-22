# Plant Disease Detection — Kerala

Detects okra and brassica leaf diseases from smartphone photographs.
Designed for farmers in Kerala, South India.

## Supported diseases

- Okra: Yellow Vein Mosaic Virus, Powdery Mildew, Cercospora Leaf Spot, Enation Leaf Curl
- Brassica (broccoli/cabbage): Black Rot, Downy Mildew, Alternaria Leaf Spot, Clubroot

## Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/plant-disease-kerala.git
cd plant-disease-kerala

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate.bat       # Windows CMD
# source venv/Scripts/activate  # Git Bash

# 3. Copy and fill environment template
cp .env.template .env
# Edit .env with your Kaggle, GitHub, and Wandb credentials

# 4. Run pipeline (trains the model — takes ~4 hours)
python run_pipeline.py
```

## Starting the server

```bash
# Development (auto-reload on code change)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production (4 workers, no reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Open browser at: http://localhost:8000

## Adding Kerala field images for tier-3 evaluation

```bash
python tools/add_kerala_image.py --path path/to/image.jpg --class okra_yvmv
```

## Running evaluations manually

```bash
# Tier-2 PlantDoc evaluation (ONCE — run after training complete)
python training/08_evaluate_tier2_plantdoc.py

# Local test set evaluation (after tier-2)
python training/10_evaluate_local_test.py

# Tier-3 Kerala evaluation (when 50+ Kerala images collected)
python training/09_evaluate_tier3_kerala.py
```

## Pipeline steps

| Step | Script | What it does |
|------|--------|--------------|
| 0 | setup/setup_project.py | Creates directories, validates env |
| 1 | setup/install_cuda.py | CUDA 12.1 installation guide |
| 2 | setup/install_dependencies.py | Installs Python packages |
| 3 | agents/download_orchestrator.py | Downloads 6 training datasets |
| 4 | agents/acquire_kerala_images.py | iNaturalist + YouTube + synthetic |
| 5 | training/01_prepare_data.py | Label assertions, split, source_map.csv |
| 6 | training/02_generate_severity.py | Severity proxy labels |
| 7 | training/03_cache_features.py | Backbone feature caching |
| 8 | training/04_train_phase1.py | Head training (~30 min) |
| 9 | training/05_train_phase2.py | Full fine-tuning (~3.5 hr) |
| 10 | training/06_calibrate.py | Temperature scaling |
| 11 | training/07_evaluate_validation.py | Validation report |
| 12 | setup/test_server.py | Server smoke test |
| 13 | training/08_evaluate_tier2_plantdoc.py | Tier-2 evaluation (ONCE) |
| 14 | setup/package_deployment.py | Dockerfile |
