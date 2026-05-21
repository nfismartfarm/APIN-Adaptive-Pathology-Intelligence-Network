"""Assemble the Hugging Face Space bundle for dxv-404/Apin.

Stages ONLY the application code (Python + HTML) into deploy/apin-space/.
The ~722 MB of model weights are deliberately left out — the Dockerfile
pulls them at build time from the dxv-404/apin-models repo. The working
tree is 50+ GB of training data and checkpoints; this keeps the Space's
git footprint to a few MB of source.

What is staged:
    Dockerfile, README.md, requirements.txt, .dockerignore, download_weights.py
    scripts/__init__.py                         (created — package marker)
    scripts/models.py
    scripts/apin/            *.py *.html         (NOT caches/ — from apin-models)
    scripts/apin_v2/         *.py *.html
    scripts/ladi_net/        *.py
    scripts/model3_training/ *.py                (NOT checkpoints/ — from apin-models)
    scripts/dinov2_probe/    *.py                (NOT results/ — from apin-models)
    scripts/psv/             *.py

Run:  python deploy/stage_space.py
Then: hf upload dxv-404/Apin deploy/apin-space . --repo-type space
"""
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "deploy", "apin-space")
SPACE_FILES = os.path.join(ROOT, "deploy", "space_files")

# Code packages the import graph needs (traced from apin_server.py) and the
# set of file extensions to copy from each. Anything else (logs, result JSONs,
# pickles, checkpoints) is either runtime history or comes from apin-models.
CODE_PACKAGES = {
    "scripts/apin":            {".py", ".html"},
    "scripts/apin_v2":         {".py", ".html"},
    "scripts/ladi_net":        {".py"},
    "scripts/model3_training": {".py"},
    "scripts/dinov2_probe":    {".py"},
    "scripts/psv":             {".py"},
}
SINGLE_FILES = [
    "scripts/models.py",
    # app/* config modules imported by the Model 2 / Model 3 / router code
    # (`from app.config_model2/3/router import ...`). Only these are needed —
    # the rest of app/ is the unrelated okra/brassica FastAPI app and would
    # drag in a second model stack, so it is deliberately left out.
    "app/__init__.py",
    "app/config_model2.py",
    "app/config_model3.py",
    "app/config_router.py",
]

# Directory names never copied — pycache, test cache, and the apin signal
# caches (caches/ is restored from apin-models at build time).
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "caches"}


def _is_dev_helper(fname):
    """Dev-only ad-hoc scripts never shipped to the Space — the `_*test*.py`
    and `_*seed*.py` helpers. They are not imported by the server."""
    b = fname.lower()
    return b.startswith("_") and ("test" in b or "seed" in b)

# Static deployment artifacts copied verbatim into the bundle root.
ROOT_FILES = ["Dockerfile", "README.md", "requirements.txt",
              ".dockerignore", "download_weights.py"]


def stage_package(rel_pkg, exts, counters):
    """Copy files of the given extensions from one package tree."""
    src_root = os.path.join(ROOT, rel_pkg)
    if not os.path.isdir(src_root):
        print(f"  MISSING package: {rel_pkg}")
        counters["missing"].append(rel_pkg)
        return
    n, nbytes = 0, 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if os.path.splitext(fname)[1] not in exts:
                continue
            if _is_dev_helper(fname):
                continue
            src = os.path.join(dirpath, fname)
            rel = os.path.relpath(src, ROOT)
            dst = os.path.join(STAGE, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
            nbytes += os.path.getsize(src)
    counters["files"] += n
    counters["bytes"] += nbytes
    print(f"  {rel_pkg:28s}  {n:4d} files  {nbytes/1024:8.1f} KB")


def main():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE, exist_ok=True)

    counters = {"files": 0, "bytes": 0, "missing": []}

    print("=" * 64)
    print("Staging the HF Space bundle -> deploy/apin-space/")
    print("=" * 64)

    # 1. Root deployment artifacts.
    for fname in ROOT_FILES:
        src = os.path.join(SPACE_FILES, fname)
        if not os.path.exists(src):
            print(f"  MISSING space file: {fname}")
            counters["missing"].append(f"space_files/{fname}")
            continue
        shutil.copy2(src, os.path.join(STAGE, fname))
    print(f"  root artifacts              {len(ROOT_FILES):4d} files  "
          f"(Dockerfile, README, requirements, ...)")

    # 1b. Project logo — the weekly PDF report renders it on the cover.
    #     report_pdf.py reads it from the bundle root (_ROOT/logo.png).
    logo_src = os.path.join(ROOT, "logo.png")
    if os.path.exists(logo_src):
        shutil.copy2(logo_src, os.path.join(STAGE, "logo.png"))
        print(f"  logo.png                       1 files  "
              f"{os.path.getsize(logo_src)/1024:8.1f} KB")
    else:
        print("  WARNING: logo.png not at project root — the report cover "
              "will render without a logo (handled gracefully).")

    # 2. scripts/ package marker — make `scripts` an explicit package so
    #    `python scripts/apin_v2/apin_server.py` resolves `import scripts.*`
    #    deterministically regardless of namespace-package behaviour.
    os.makedirs(os.path.join(STAGE, "scripts"), exist_ok=True)
    with open(os.path.join(STAGE, "scripts", "__init__.py"), "w",
              encoding="utf-8") as f:
        f.write("# APIN — scripts package marker.\n")

    # 3. Single top-level modules.
    for rel in SINGLE_FILES:
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            print(f"  MISSING file: {rel}")
            counters["missing"].append(rel)
            continue
        dst = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        counters["files"] += 1
        counters["bytes"] += os.path.getsize(src)

    # 4. Code packages.
    for rel_pkg, exts in CODE_PACKAGES.items():
        stage_package(rel_pkg, exts, counters)

    # 5. Runtime-writable dir the apin feedback endpoint appends to.
    feedback_dir = os.path.join(STAGE, "scripts", "apin", "feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    with open(os.path.join(feedback_dir, ".gitkeep"), "w") as f:
        f.write("")

    print("-" * 64)
    print(f"  TOTAL code staged: {counters['files']} files  "
          f"{counters['bytes']/1024/1024:.2f} MB")
    print(f"  Location:          {STAGE}")
    if counters["missing"]:
        print()
        print(f"  WARNING — {len(counters['missing'])} missing item(s):")
        for m in counters["missing"]:
            print(f"      {m}")
        sys.exit(1)
    print()
    print("  Bundle ready. Next:")
    print("    hf upload dxv-404/Apin deploy/apin-space . --repo-type space")
    sys.exit(0)


if __name__ == "__main__":
    main()
