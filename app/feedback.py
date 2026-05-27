# app/feedback.py
"""Active learning feedback collection and analytics."""
import json
import os
import threading
from datetime import datetime
from app.config import ROOT, CLASS_NAMES

FEEDBACK_FILE = os.path.join(ROOT, 'data', 'feedback_db.json')
_feedback_lock = threading.Lock()


def _load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return {'entries': [], 'summary': {}}
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {'entries': [], 'summary': {}}


def _save_feedback(db):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(db, f, indent=2)


def record_feedback(predicted_class, correct_class, confidence,
                     uncertainty, is_correction, image_hash=None):
    entry = {
        'timestamp': datetime.now().isoformat(),
        'predicted_class': predicted_class,
        'correct_class': correct_class,
        'is_correction': is_correction,
        'confidence': round(float(confidence), 4),
        'uncertainty': round(float(uncertainty), 4),
        'image_hash': image_hash,
    }
    with _feedback_lock:
        db = _load_feedback()
        db['entries'].append(entry)
        if is_correction and correct_class:
            key = f'{predicted_class} -> {correct_class}'
            db['summary'][key] = db['summary'].get(key, 0) + 1
        _save_feedback(db)
    return entry


def get_feedback_analytics():
    with _feedback_lock:
        db = _load_feedback()
    entries = db.get('entries', [])
    if not entries:
        return {'total_entries': 0, 'corrections': 0, 'confirmations': 0,
                'correction_rate': 0.0, 'top_confusions': [],
                'per_class_corrections': {}}
    corrections = [e for e in entries if e.get('is_correction')]
    confirmations = [e for e in entries if not e.get('is_correction')]
    per_class = {}
    for e in corrections:
        cls = e.get('predicted_class', 'unknown')
        per_class[cls] = per_class.get(cls, 0) + 1
    summary = db.get('summary', {})
    top_confusions = sorted(summary.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        'total_entries': len(entries),
        'corrections': len(corrections),
        'confirmations': len(confirmations),
        'correction_rate': round(len(corrections) / max(len(entries), 1), 3),
        'top_confusions': [{'confusion': k, 'count': v} for k, v in top_confusions],
        'per_class_corrections': per_class,
    }


def get_all_feedback_entries(limit=100):
    with _feedback_lock:
        db = _load_feedback()
    entries = db.get('entries', [])
    return sorted(entries, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
