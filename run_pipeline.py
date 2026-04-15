# run_pipeline.py
"""
Automated pipeline runner for the plant disease detection project.

Usage:
    python run_pipeline.py                    # run from step 0
    python run_pipeline.py --from-step 5      # resume from step 5
    python run_pipeline.py --step 8           # run only step 8
    python run_pipeline.py --reset-step 8     # mark step 8 as incomplete
    python run_pipeline.py --status           # show step completion status
    python run_pipeline.py --yes              # no interactive prompts

[FIX GAP 21] Step execution rules:
    String steps  (e.g. "setup/setup_project.py") → subprocess.run([sys.executable, path, ...])
    Lambda steps  (e.g. lambda: run_downloads())  → called directly in-process
[FIX GAP 51] --yes is propagated to all subprocesses (install_cuda.py, etc.)
[FIX GAP 65] load_dotenv() called before any step runs
[FIX GAP 66] WANDB_MODE=offline set as fallback if WANDB_API_KEY missing
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# [FIX GAP 66] WANDB offline fallback
if not os.environ.get('WANDB_API_KEY'):
    os.environ.setdefault('WANDB_MODE', 'offline')

ROOT          = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(ROOT, '.pipeline_progress.json')
LOG_FILE      = os.path.join(ROOT, 'pipeline_failures.log')


# ── Step definitions ───────────────────────────────────────────────────────
# [FIX GAP 21] String = subprocess, lambda = direct call.
# Lambdas used only for steps that need in-process module access
# (download orchestrator, acquisition) where subprocess pickling is complex.

def _make_steps(yes_flag):
    """Returns STEPS list. yes_flag is passed to subprocess calls."""
    yes_args = ['--yes'] if yes_flag else []

    def _run_downloads():
        sys.path.insert(0, ROOT)
        from agents.download_orchestrator import run_all_downloads
        return run_all_downloads()

    def _acquire_kerala():
        sys.path.insert(0, ROOT)
        from agents.acquire_kerala_images import acquire_all
        return acquire_all()

    STEPS = [
        # Step 0: environment setup
        'setup/setup_project.py',
        # Step 1: CUDA installation — [FIX GAP 51] pass --yes if set
        ['setup/install_cuda.py'] + yes_args,
        # Step 2: dependency installation
        'setup/install_dependencies.py',
        # Step 3: dataset downloads (lambda — in-process parallel)
        lambda: _run_downloads(),
        # Step 4: Kerala image acquisition (lambda — in-process parallel)
        lambda: _acquire_kerala(),
        # Step 5: data preparation and source_map.csv
        'training/01_prepare_data.py',
        # Step 6: severity proxy labels
        'training/02_generate_severity.py',
        # Step 7: feature caching
        'training/03_cache_features.py',
        # Step 8: Phase 1 training (heads only)
        'training/04_train_phase1.py',
        # Step 9: Phase 2 training (full fine-tuning)
        'training/05_train_phase2.py',
        # Step 10: temperature calibration
        'training/06_calibrate.py',
        # Step 11: validation report
        'training/07_evaluate_validation.py',
        # Step 12: server smoke test
        'setup/test_server.py',
        # Step 13: tier-2 PlantDoc evaluation (ONCE) — [FIX GAP 54] pass --yes
        ['training/08_evaluate_tier2_plantdoc.py'] + yes_args,
        # Step 14: deployment packaging
        'setup/package_deployment.py',
    ]
    return STEPS


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def is_done(step_idx, progress):
    return progress.get(str(step_idx), {}).get('done', False)


def mark_done(step_idx, progress):
    progress[str(step_idx)] = {
        'done'       : True,
        'timestamp'  : datetime.datetime.now().isoformat(),
    }
    save_progress(progress)


def mark_undone(step_idx, progress):
    if str(step_idx) in progress:
        progress[str(step_idx)]['done'] = False
        save_progress(progress)


def run_smoke_test(step_idx):
    """
    Per-step smoke test. Returns True if passes, False if fails.
    [FIX GAP 23] Step 11 smoke test added.
    """
    tests = {
        0  : lambda: os.path.exists(os.path.join(ROOT, 'data', 'metadata')),
        1  : _smoke_cuda,
        2  : _smoke_imports,
        3  : _smoke_downloads,
        5  : lambda: os.path.exists(os.path.join(ROOT, 'data', 'metadata', 'source_map.csv')),
        6  : lambda: os.path.exists(os.path.join(ROOT, 'data', 'metadata', 'severity_labels.csv')),
        7  : lambda: (
            os.path.exists(os.path.join(ROOT, 'cache', 'train_features.pt')) and
            os.path.exists(os.path.join(ROOT, 'cache', 'val_features.pt'))
        ),
        8  : lambda: os.path.exists(os.path.join(ROOT, 'models', 'checkpoints', 'phase1_best.pt')),
        9  : lambda: os.path.exists(os.path.join(ROOT, 'models', 'best_model.pt')),
        10 : lambda: os.path.exists(os.path.join(ROOT, 'models', 'temperature.pt')),
        # [FIX GAP 23] Step 11 smoke test:
        11 : lambda: any(
            f.endswith('.md') for f in os.listdir(os.path.join(ROOT, 'reports'))
        ) if os.path.isdir(os.path.join(ROOT, 'reports')) else False,
        12 : _smoke_server,
        13 : lambda: any(
            f.startswith('tier2') and f.endswith('.md')
            for f in os.listdir(os.path.join(ROOT, 'reports'))
        ) if os.path.isdir(os.path.join(ROOT, 'reports')) else False,
        14 : lambda: os.path.exists(os.path.join(ROOT, 'Dockerfile')),
    }
    test_fn = tests.get(step_idx)
    if test_fn is None:
        return True  # No smoke test for this step
    try:
        result = test_fn()
        if result:
            print(f"  [OK] Smoke test for Step {step_idx} passed")
        else:
            print(f"  [FAIL] Smoke test for Step {step_idx} FAILED")
        return result
    except Exception as e:
        print(f"  [FAIL] Smoke test for Step {step_idx} raised exception: {e}")
        return False


def _smoke_cuda():
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch; assert torch.cuda.is_available(), "CUDA not available"'],
        capture_output=True, text=True
    )
    return result.returncode == 0


def _smoke_imports():
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch, timm, albumentations, sklearn, cv2, fastapi, wandb'],
        capture_output=True, text=True
    )
    return result.returncode == 0


def _smoke_downloads():
    raw_dir    = os.path.join(ROOT, 'data', 'raw')
    results_f  = os.path.join(ROOT, 'data', 'metadata', 'download_results.json')
    if not os.path.exists(results_f):
        return False
    with open(results_f) as f:
        results = json.load(f)
    # Minimum: gadde_okra and cabbage_balanced must have images
    mandatories = ['gadde_okra', 'cabbage_balanced']
    for name in mandatories:
        r = results.get(name, {})
        if not r.get('success', False):
            print(f"  Mandatory dataset {name} failed download: {r.get('error')}")
            return False
    return True


def _smoke_server():
    """
    [FIX GAP 5] Test image has texture (noise overlay) so it passes blur check.
    """
    import time
    import threading

    server_proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'app.main:app',
         '--host', '0.0.0.0', '--port', '8765'],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'setup', 'test_server.py')],
            capture_output=True, text=True, timeout=30
        )
        passed = result.returncode == 0
        if not passed:
            print(f"  Server test stdout: {result.stdout[-500:]}")
            print(f"  Server test stderr: {result.stderr[-500:]}")
        return passed
    except Exception as e:
        print(f"  Server smoke test failed: {e}")
        return False
    finally:
        server_proc.terminate()


def github_commit(step_idx):
    """Commit and push after each step. Retry 3x. Log failures. Never stop pipeline."""
    msg = f"Step {step_idx:02d} complete — automated pipeline commit"
    cmds = [
        ['git', 'add', '-A'],
        ['git', 'commit', '-m', msg, '--allow-empty'],
        ['git', 'push'],
    ]
    for cmd in cmds:
        for attempt in range(3):
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                break
            if attempt == 2:
                commit_failure_log(step_idx, cmd, r)


def commit_failure_log(step_idx, cmd, result):
    with open(LOG_FILE, 'a') as f:
        f.write(
            f"[{datetime.datetime.now().isoformat()}] "
            f"Step {step_idx} git failure: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout[:200]}\n"
            f"  stderr: {result.stderr[:200]}\n"
        )


def execute_step(step, step_idx, yes_flag):
    """
    [FIX GAP 21] String steps → subprocess.run with sys.executable.
    List steps → subprocess.run with the list items.
    Lambda steps → called directly in-process.
    """
    if callable(step):
        # Lambda: call directly
        result = step()
        print(f"  Step {step_idx} (in-process) returned: {result}")
    elif isinstance(step, list):
        # List: first item is script path, rest are args
        script_path = os.path.join(ROOT, step[0])
        extra_args  = step[1:]
        cmd = [sys.executable, script_path] + extra_args
        r   = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(
                f"Step {step_idx} subprocess failed with returncode {r.returncode}: "
                f"{' '.join(cmd)}"
            )
    else:
        # String: script path relative to ROOT
        script_path = os.path.join(ROOT, step)
        cmd = [sys.executable, script_path]
        r   = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(
                f"Step {step_idx} subprocess failed with returncode {r.returncode}: "
                f"{' '.join(cmd)}"
            )


def show_status(progress, steps):
    print(f"\nPipeline status ({len(steps)} steps):")
    for i, step in enumerate(steps):
        done   = is_done(i, progress)
        status = '[OK]' if done else '[ ]'
        ts     = progress.get(str(i), {}).get('timestamp', '')
        label  = step.__name__ if callable(step) else (step[0] if isinstance(step, list) else step)
        print(f"  [{status}] Step {i:2d}: {label}  {ts}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Run the plant disease pipeline')
    parser.add_argument('--from-step', type=int, default=0,
                        help='Start from this step (skip completed)')
    parser.add_argument('--step', type=int, default=None,
                        help='Run only this specific step')
    parser.add_argument('--reset-step', type=int, default=None,
                        help='Mark a step as incomplete')
    parser.add_argument('--status', action='store_true',
                        help='Show completion status and exit')
    parser.add_argument('--yes', action='store_true',
                        help='Pass --yes to all subprocesses (no interactive prompts)')
    args = parser.parse_args()

    steps    = _make_steps(args.yes)
    progress = load_progress()

    if args.status:
        show_status(progress, steps)
        return

    if args.reset_step is not None:
        mark_undone(args.reset_step, progress)
        print(f"Step {args.reset_step} marked as incomplete.")
        return

    # Determine which steps to run
    if args.step is not None:
        run_indices = [args.step]
    else:
        run_indices = list(range(args.from_step, len(steps)))

    print(f"Pipeline starting. Steps to run: {run_indices}")
    print(f"Platform: {sys.platform}")

    for i in run_indices:
        if i >= len(steps):
            print(f"Step {i} does not exist (max is {len(steps) - 1})")
            break

        step = steps[i]
        label = step.__name__ if callable(step) else (
            step[0] if isinstance(step, list) else step
        )

        # Skip if already done (unless --step used for single-step mode)
        if args.step is None and is_done(i, progress):
            print(f"[SKIP] Step {i:2d}: {label} (already done)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[RUN ] Step {i:2d}: {label}")
        print(f"{'=' * 60}")

        try:
            execute_step(step, i, args.yes)
            smoke_passed = run_smoke_test(i)
            if not smoke_passed:
                print(f"WARNING: smoke test failed for step {i}. Continuing anyway.")
            mark_done(i, progress)
            github_commit(i)
            print(f"[DONE] Step {i:2d}: {label}")
        except Exception as e:
            traceback.print_exc()
            print(f"\n[FAIL] Step {i:2d}: {label}")
            print(f"Error: {e}")
            with open(LOG_FILE, 'a') as f:
                f.write(
                    f"\n[{datetime.datetime.now().isoformat()}] "
                    f"Step {i} FAILED: {e}\n"
                    f"{traceback.format_exc()}\n"
                )
            print(f"Failure logged to {LOG_FILE}")
            print(f"To retry: python run_pipeline.py --step {i}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Pipeline complete.")
    show_status(progress, steps)


if __name__ == '__main__':
    main()
