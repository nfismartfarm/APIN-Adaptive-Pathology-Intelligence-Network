# training/report.py
"""Shared report writing utilities for evaluation scripts."""

import os
import datetime


def write_report(metrics, path, title):
    """
    Writes a Markdown report from a metrics dict.
    metrics: dict of metric_name -> value
    path: output file path
    title: report title string
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f'# {title}',
        f'Generated: {datetime.datetime.now().isoformat()}',
        '',
        '## Metrics',
        '',
        '| Metric | Value |',
        '|--------|-------|',
    ]
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f'| {k} | {v:.4f} |')
        else:
            lines.append(f'| {k} | {v} |')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report written: {path}")
    return path
