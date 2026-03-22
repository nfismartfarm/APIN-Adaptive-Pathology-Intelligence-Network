# training/helpers.py
# Shared utilities for Phase 1 and Phase 2 training scripts.
# Import these at MODULE LEVEL in training scripts — NOT inside __main__.

import os
import glob
import torch


class EarlyStopping:
    """
    Monitors val/macro_f1. Stops training when score does not improve
    by min_delta for `patience` consecutive epochs.
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics, path):
    """Saves full training state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch'            : epoch,
        'model_state_dict' : model.state_dict(),
        'val_metrics'      : val_metrics,
    }
    if optimizer: state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler: state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler:    state['scaler_state_dict']    = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    """Loads training state from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('val_metrics', {})


def cleanup_old_checkpoints(ckpt_dir, keep_n=3, phase='phase1'):
    """Keeps only `keep_n` most recent epoch checkpoints per phase."""
    pattern = os.path.join(ckpt_dir, f'{phase}_epoch*.pt')
    ckpts   = sorted(glob.glob(pattern))
    to_del  = ckpts[:-keep_n] if len(ckpts) > keep_n else []
    for c in to_del:
        os.remove(c)


def get_llrd_optimizer(model, base_lr, decay=0.85, weight_decay=1e-4):
    """AdamW with Layer-wise Learning Rate Decay (LLRD)."""
    param_groups = []

    for name, module in [
        ('disease_head',    model.disease_head),
        ('severity_head',   model.severity_head),
        ('crop_classifier', model.crop_classifier),
        ('fpn',             model.fpn),
    ]:
        param_groups.append({
            'params': list(module.parameters()),
            'lr'    : base_lr,
            'name'  : name,
        })

    blocks = model._get_backbone_blocks()
    for i, block in enumerate(reversed(blocks)):
        lr = base_lr * (decay ** i)
        param_groups.append({
            'params': list(block.parameters()),
            'lr'    : lr,
            'name'  : f'backbone_block_{len(blocks) - 1 - i}',
        })

    stem = model._get_stem_params()
    if stem:
        param_groups.append({
            'params': stem,
            'lr'    : base_lr * (decay ** len(blocks)),
            'name'  : 'backbone_stem',
        })

    for pg in param_groups:
        print(f"  LLRD {pg['name']:35s}: lr={pg['lr']:.2e}")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
