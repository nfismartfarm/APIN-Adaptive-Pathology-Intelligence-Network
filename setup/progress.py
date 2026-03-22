# setup/progress.py
"""Pipeline progress tracking utilities."""

import os
import json
import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS_FILE = os.path.join(ROOT, '.pipeline_progress.json')


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def mark_step_done(step_idx):
    progress = load_progress()
    progress[str(step_idx)] = {
        'done': True,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    save_progress(progress)


def is_step_done(step_idx):
    progress = load_progress()
    return progress.get(str(step_idx), {}).get('done', False)


def print_status():
    progress = load_progress()
    print("Pipeline Progress:")
    for i in range(15):
        done = progress.get(str(i), {}).get('done', False)
        ts   = progress.get(str(i), {}).get('timestamp', '')
        mark = 'V' if done else 'O'
        print(f"  [{mark}] Step {i:2d}  {ts}")


if __name__ == '__main__':
    print_status()
