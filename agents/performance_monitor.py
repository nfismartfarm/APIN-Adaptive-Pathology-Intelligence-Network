# agents/performance_monitor.py
"""Performance monitoring utilities for training and inference."""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PerformanceMonitor:
    """Tracks and logs performance metrics during pipeline execution."""

    def __init__(self, log_path=None):
        from app.config import REPORTS
        self.log_path = log_path or os.path.join(REPORTS, 'performance_log.json')
        self.entries = []

    def start_timer(self, step_name):
        """Start timing a pipeline step."""
        return {'step': step_name, 'start': time.time()}

    def end_timer(self, timer):
        """End timing and record the result."""
        elapsed = time.time() - timer['start']
        entry = {
            'step': timer['step'],
            'elapsed_seconds': round(elapsed, 2),
            'elapsed_human': f"{elapsed/60:.1f} min" if elapsed > 60 else f"{elapsed:.1f} sec",
        }
        self.entries.append(entry)
        print(f"  [{timer['step']}] completed in {entry['elapsed_human']}")
        return entry

    def save(self):
        """Save all entries to JSON log."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def gpu_memory_usage(self):
        """Return current GPU memory usage if CUDA available."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = total - free
                return {
                    'used_gb': round(used / 1e9, 2),
                    'total_gb': round(total / 1e9, 2),
                    'free_gb': round(free / 1e9, 2),
                }
        except Exception:
            pass
        return {'used_gb': 0, 'total_gb': 0, 'free_gb': 0}
