#!/usr/bin/env python
"""
Unified training script for TACTIC-Kinetics.

Supports all model versions:
- v0: Single-curve baseline (22% accuracy)
- v1: Basic multi-condition, 5 conditions, 2 features (57%)
- v2: Improved multi-condition, 20 conditions, 5 features, derived, pairwise (65%)
- v3: Multi-task learning with auxiliary heads (66%)

Usage:
    python train.py --version v0 --epochs 100
    python train.py --version v1 --epochs 100
    python train.py --version v2 --epochs 100
    python train.py --version v3 --epochs 100 --multi-task
"""

import os
import sys
import csv
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))

from tactic_kinetics.models.multi_condition_classifier import (
    SingleCurveClassifier,
    BasicMultiConditionClassifier,
    MultiConditionClassifier,
    MultiTaskClassifier,
    create_single_curve_model,
    create_basic_multi_condition_model,
    create_multi_condition_model,
    create_multi_task_model,
)
from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    load_dataset,
    save_dataset,
)
from tactic_kinetics.training.multi_condition_dataset import (
    MultiConditionDataset,
    MultiConditionDatasetConfig,
    multi_condition_collate_fn,
)
from multiprocessing import cpu_count


# =============================================================================
# VERSION CONFIGURATIONS
# =============================================================================

VERSION_CONFIGS = {
    'v0': {
        'description': 'Single-curve baseline',
        'n_conditions': 1,
        'n_traj_features': 2,  # time, substrate
        'n_condition_features': 4,  # T, pH, S0, E0
        'use_derived': False,
        'use_pairwise': False,
        'multi_task': False,
        'dataset_path': 'data/synthetic_dataset.pt',
        'expected_accuracy': '~22%',
    },
    'v1': {
        'description': 'Basic multi-condition (5 cond, 2 features)',
        'n_conditions': 5,
        'n_traj_features': 2,  # time, substrate
        'n_condition_features': 6,
        'use_derived': False,
        'use_pairwise': False,
        'multi_task': False,
        'dataset_path': 'data/multi_condition_v1.pt',
        'expected_accuracy': '~57%',
    },
    'v2': {
        'description': 'Improved multi-condition (20 cond, 5 features, derived, pairwise)',
        'n_conditions': 20,
        'n_traj_features': 5,  # time, S, P, dS/dt, dP/dt
        'n_condition_features': 8,
        'use_derived': True,
        'use_pairwise': True,
        'multi_task': False,
        'dataset_path': 'data/multi_condition_v2.pt',
        'expected_accuracy': '~65%',
    },
    'v3': {
        'description': 'Multi-task with auxiliary heads',
        'n_conditions': 20,
        'n_traj_features': 5,
        'n_condition_features': 8,
        'use_derived': True,
        'use_pairwise': True,
        'multi_task': True,
        'dataset_path': 'data/multi_condition_v3.pt',
        'expected_accuracy': '~66%',
    },
}

MECHANISM_NAMES = [
    'michaelis_menten_irreversible',
    'michaelis_menten_reversible',
    'competitive_inhibition',
    'uncompetitive_inhibition',
    'mixed_inhibition',
    'substrate_inhibition',
    'ordered_bi_bi',
    'random_bi_bi',
    'ping_pong',
    'product_inhibition',
]


# =============================================================================
# V0 DATASET (Single-Curve)
# =============================================================================

class SingleCurveDataset(Dataset):
    """Dataset for v0 single-curve training."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'times': sample['times'],
            'values': sample['values'],
            'conditions': sample['conditions'],
            'mask': sample['mask'],
            'mechanism_idx': torch.tensor(sample['mechanism_idx'], dtype=torch.long),
        }


def single_curve_collate_fn(batch):
    """Collate function for single-curve data."""
    return {
        'times': torch.stack([b['times'] for b in batch]),
        'values': torch.stack([b['values'] for b in batch]),
        'conditions': torch.stack([b['conditions'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'mechanism_idx': torch.stack([b['mechanism_idx'] for b in batch]),
    }


# =============================================================================
# V1 DATASET (Basic Multi-Condition with 2 features)
# =============================================================================

class BasicMultiConditionDataset(Dataset):
    """Dataset for v1 basic multi-condition training (2 trajectory features)."""

    def __init__(self, samples: list, max_conditions: int = 5, n_timepoints: int = 20):
        self.samples = samples
        self.max_conditions = max_conditions
        self.n_timepoints = n_timepoints
        self._compute_stats()

    def _compute_stats(self):
        """Compute normalization statistics."""
        all_conc = []
        all_times = []

        for sample in self.samples:
            for traj in sample.trajectories:
                all_times.extend(traj['t'].tolist())
                for species, conc in traj['concentrations'].items():
                    all_conc.extend(conc.tolist())

        self.conc_mean = np.mean(all_conc)
        self.conc_std = np.std(all_conc) + 1e-8
        self.time_max = np.max(all_times) + 1e-8

        print(f"Dataset stats: conc_mean={self.conc_mean:.4f}, conc_std={self.conc_std:.4f}, time_max={self.time_max:.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        trajectories = []
        conditions_list = []

        for traj_data in sample.trajectories[:self.max_conditions]:
            concentrations = traj_data['concentrations']
            t = traj_data['t']
            conditions = traj_data['conditions']

            # Get substrate (S for single, A for bi-substrate)
            if 'S' in concentrations:
                S = concentrations['S']
            elif 'A' in concentrations:
                S = concentrations['A']
            else:
                S = list(concentrations.values())[0]

            # Interpolate to fixed timepoints
            if len(t) != self.n_timepoints:
                t_new = np.linspace(t[0], t[-1], self.n_timepoints)
                S = np.interp(t_new, t, S)
                t = t_new

            # Normalize (only time and substrate for v1)
            t_norm = t / self.time_max
            S_norm = (S - self.conc_mean) / self.conc_std

            # Stack trajectory features: (n_timepoints, 2)
            traj_features = np.stack([t_norm, S_norm], axis=-1)
            trajectories.append(traj_features)

            # Condition features (6 features for v1)
            cond_features = self._extract_conditions_v1(conditions)
            conditions_list.append(cond_features)

        # Pad to max_conditions
        n_actual = len(trajectories)
        n_pad = self.max_conditions - n_actual

        if n_pad > 0:
            pad_traj = np.zeros((self.n_timepoints, 2))
            pad_cond = np.zeros(6)
            for _ in range(n_pad):
                trajectories.append(pad_traj)
                conditions_list.append(pad_cond)

        condition_mask = np.array([False] * n_actual + [True] * n_pad)

        return {
            'trajectories': torch.tensor(np.stack(trajectories), dtype=torch.float32),
            'conditions': torch.tensor(np.stack(conditions_list), dtype=torch.float32),
            'condition_mask': torch.tensor(condition_mask, dtype=torch.bool),
            'mechanism_idx': torch.tensor(sample.mechanism_idx, dtype=torch.long),
        }

    def _extract_conditions_v1(self, conditions: dict) -> np.ndarray:
        """Extract 6 condition features for v1."""
        features = np.zeros(6)

        # S0 or A0
        S0 = conditions.get('S0', conditions.get('A0', 1.0))
        features[0] = np.log10(max(S0, 1e-9))

        # I0 or B0
        I0 = conditions.get('I0', conditions.get('B0', 0))
        features[1] = np.log10(max(I0, 1e-9)) if I0 > 0 else -9.0

        # E0
        E0 = conditions.get('E0', 1e-3)
        features[2] = np.log10(max(E0, 1e-12))

        # Temperature
        T = conditions.get('T', 298.15)
        features[3] = (T - 298.15) / 20.0

        # pH
        pH = conditions.get('pH', 7.0)
        features[4] = (pH - 7.0) / 2.0

        # P0
        P0 = conditions.get('P0', 0)
        features[5] = np.log10(max(P0, 1e-9)) if P0 > 0 else -9.0

        return features


def basic_multi_condition_collate_fn(batch):
    """Collate function for v1 basic multi-condition data."""
    return {
        'trajectories': torch.stack([b['trajectories'] for b in batch]),
        'conditions': torch.stack([b['conditions'] for b in batch]),
        'condition_mask': torch.stack([b['condition_mask'] for b in batch]),
        'mechanism_idx': torch.stack([b['mechanism_idx'] for b in batch]),
    }


# =============================================================================
# TRAINING LOGGER
# =============================================================================

class TrainingLogger:
    """Handles all logging during training."""

    def __init__(self, log_dir: Path, log_interval: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.global_step = 0

        # Initialize CSV batch log
        self.batch_log_path = self.log_dir / "batch_log.csv"
        self._init_batch_log()

        # History for JSON export
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_grad_norm_mean': [],
            'epoch_grad_norm_max': [],
            'train_cls_loss': [],
            'train_aux_loss': [],
            'val_cls_loss': [],
            'val_kinetic_mse': [],
            'val_pattern_mse': [],
            'aux_weight': [],
            'val_per_mechanism_acc': [],
        }

    def _init_batch_log(self):
        """Initialize CSV file for batch-level logging."""
        with open(self.batch_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step", "epoch", "batch", "loss", "accuracy",
                "grad_norm", "learning_rate", "nan_count", "batch_size"
            ])

    def log_batch(self, epoch: int, batch: int, metrics: dict):
        """Log batch metrics to CSV."""
        self.global_step += 1

        with open(self.batch_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.global_step,
                epoch,
                batch,
                f"{metrics['loss']:.6f}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['grad_norm']:.4f}",
                f"{metrics['learning_rate']:.8f}",
                metrics.get('nan_count', 0),
                metrics['batch_size'],
            ])

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float,
                  aux_weight: float = 0.0):
        """Log epoch-level metrics."""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['learning_rate'].append(lr)
        self.history['epoch_grad_norm_mean'].append(train_metrics.get('grad_norm_mean', 0))
        self.history['epoch_grad_norm_max'].append(train_metrics.get('grad_norm_max', 0))
        self.history['train_cls_loss'].append(train_metrics.get('cls_loss', train_metrics['loss']))
        self.history['train_aux_loss'].append(train_metrics.get('aux_loss', 0))
        self.history['val_cls_loss'].append(val_metrics.get('cls_loss', val_metrics['loss']))
        self.history['val_kinetic_mse'].append(val_metrics.get('kinetic_mse', 0))
        self.history['val_pattern_mse'].append(val_metrics.get('pattern_mse', 0))
        self.history['aux_weight'].append(aux_weight)
        self.history['val_per_mechanism_acc'].append(val_metrics.get('per_class_accuracy', {}))

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_multi_task_loss(
    output: dict,
    labels: torch.Tensor,
    kinetic_params_target: torch.Tensor,
    param_pattern_target: torch.Tensor,
    condition_mask: torch.Tensor,
    aux_weight: float = 0.3,
) -> dict:
    """Compute multi-task loss for v3."""
    loss_cls = F.cross_entropy(output['logits'], labels)

    kinetic_params_target = torch.nan_to_num(kinetic_params_target, nan=0.0, posinf=10.0, neginf=-10.0)
    param_pattern_target = torch.nan_to_num(param_pattern_target, nan=0.0, posinf=10.0, neginf=-10.0)

    if 'kinetic_params' in output:
        kinetic_params_pred = output['kinetic_params']
        valid_mask = ~condition_mask
        valid_mask = valid_mask.unsqueeze(-1).expand_as(kinetic_params_pred)
        kinetic_diff = (kinetic_params_pred - kinetic_params_target) ** 2
        kinetic_diff = kinetic_diff * valid_mask.float()
        n_valid = valid_mask.float().sum()
        loss_kinetic = kinetic_diff.sum() / (n_valid + 1e-8)
        loss_kinetic = torch.clamp(loss_kinetic, 0.0, 100.0)
    else:
        loss_kinetic = torch.tensor(0.0, device=labels.device)

    if 'param_pattern' in output:
        param_pattern_pred = output['param_pattern']
        loss_pattern = F.mse_loss(param_pattern_pred, param_pattern_target)
        loss_pattern = torch.clamp(loss_pattern, 0.0, 100.0)
    else:
        loss_pattern = torch.tensor(0.0, device=labels.device)

    loss_total = loss_cls + aux_weight * (loss_kinetic + loss_pattern)

    return {
        'total': loss_total,
        'classification': loss_cls,
        'kinetic_params': loss_kinetic,
        'param_pattern': loss_pattern,
    }


def get_aux_weight(epoch: int, total_epochs: int, initial_weight: float = 0.5) -> float:
    """Curriculum learning: decrease auxiliary loss weight over training."""
    progress = epoch / max(total_epochs - 1, 1)
    return initial_weight * (1 - 0.8 * progress)


def setup_device(gpu_ids: list = None):
    """Setup device for training."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        print(f"Using GPUs: {gpu_ids}")

    device = torch.device("cuda:0")
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    return device


# =============================================================================
# V0 TRAINING FUNCTIONS
# =============================================================================

def train_epoch_v0(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger,
    grad_clip: float = 1.0,
    log_interval: int = 10,
) -> dict:
    """Train v0 (single-curve) for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    nan_batches = 0
    grad_norms = []

    for batch_idx, batch in enumerate(dataloader):
        times = batch['times'].to(device)
        values = batch['values'].to(device)
        conditions = batch['conditions'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['mechanism_idx'].to(device)
        batch_size = labels.size(0)

        optimizer.zero_grad()

        output = model(times, values, conditions, mask)
        logits = output['logits']

        loss = F.cross_entropy(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            continue

        loss.backward()

        grad_norm = compute_grad_norm(model)
        grad_norms.append(grad_norm)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        batch_acc = batch_correct / batch_size

        total_loss += loss.item()
        correct += batch_correct
        total += batch_size

        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_batch(epoch, batch_idx, {
                'loss': loss.item(),
                'accuracy': batch_acc,
                'grad_norm': grad_norm,
                'learning_rate': current_lr,
                'nan_count': nan_batches,
                'batch_size': batch_size,
            })
            print(f"  Batch {batch_idx:4d}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {batch_acc*100:5.1f}% | "
                  f"GradNorm: {grad_norm:.2f}")

    n_batches = max(len(dataloader) - nan_batches, 1)
    return {
        'loss': total_loss / n_batches,
        'accuracy': correct / max(total, 1),
        'grad_norm_mean': np.mean(grad_norms) if grad_norms else 0,
        'grad_norm_max': np.max(grad_norms) if grad_norms else 0,
        'grad_norm_std': np.std(grad_norms) if grad_norms else 0,
        'nan_batches': nan_batches,
    }


@torch.no_grad()
def validate_v0(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate v0 model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        times = batch['times'].to(device)
        values = batch['values'].to(device)
        conditions = batch['conditions'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['mechanism_idx'].to(device)

        output = model(times, values, conditions, mask)
        logits = output['logits']

        loss = F.cross_entropy(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        valid_batches += 1
        total_loss += loss.item()

        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for c in range(10):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = float((all_preds[mask] == c).mean())
        else:
            per_class_acc[c] = 0.0

    return {
        'loss': total_loss / max(valid_batches, 1),
        'accuracy': correct / max(total, 1),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'per_class_accuracy': per_class_acc,
        'mean_confidence': np.mean([p.max() for p in all_probs]) if all_probs else 0,
    }


# =============================================================================
# V1/V2/V3 TRAINING FUNCTIONS
# =============================================================================

def train_epoch_multi(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    multi_task: bool = False,
    aux_weight: float = 0.3,
    version: str = 'v2',
) -> dict:
    """Train multi-condition model (v1, v2, v3) for one epoch."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_aux_loss = 0
    correct = 0
    total = 0
    nan_batches = 0
    grad_norms = []

    for batch_idx, batch in enumerate(dataloader):
        trajectories = batch['trajectories'].to(device)
        conditions = batch['conditions'].to(device)
        condition_mask = batch['condition_mask'].to(device)
        labels = batch['mechanism_idx'].to(device)
        batch_size = labels.size(0)

        # Derived features only for v2/v3
        derived_features = batch.get('derived_features')
        if derived_features is not None:
            derived_features = derived_features.to(device)

        optimizer.zero_grad()

        # Forward pass
        if version == 'v1':
            output = model(trajectories, conditions, condition_mask=condition_mask)
        else:
            output = model(trajectories, conditions, derived_features=derived_features,
                          condition_mask=condition_mask)

        logits = output['logits']

        if multi_task and 'kinetic_params' in batch:
            kinetic_params_target = batch['kinetic_params'].to(device)
            param_pattern_target = batch['param_pattern'].to(device)

            losses = compute_multi_task_loss(
                output, labels, kinetic_params_target, param_pattern_target,
                condition_mask, aux_weight
            )
            loss = losses['total']
            cls_loss = losses['classification'].item()
            aux_loss = (losses['kinetic_params'].item() + losses['param_pattern'].item())
        else:
            loss = F.cross_entropy(logits, labels)
            cls_loss = loss.item()
            aux_loss = 0.0

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            continue

        loss.backward()

        grad_norm = compute_grad_norm(model)
        grad_norms.append(grad_norm)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        batch_acc = batch_correct / batch_size

        total_loss += loss.item()
        total_cls_loss += cls_loss
        total_aux_loss += aux_loss
        correct += batch_correct
        total += batch_size

        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']

            logger.log_batch(epoch, batch_idx, {
                'loss': loss.item(),
                'accuracy': batch_acc,
                'grad_norm': grad_norm,
                'learning_rate': current_lr,
                'nan_count': nan_batches,
                'batch_size': batch_size,
            })

            if multi_task:
                print(f"  Batch {batch_idx:4d}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} (cls:{cls_loss:.4f} aux:{aux_loss:.4f}) | "
                      f"Acc: {batch_acc*100:5.1f}% | GradNorm: {grad_norm:.2f}")
            else:
                print(f"  Batch {batch_idx:4d}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {batch_acc*100:5.1f}% | GradNorm: {grad_norm:.2f}")

    n_batches = max(len(dataloader) - nan_batches, 1)
    return {
        'loss': total_loss / n_batches,
        'cls_loss': total_cls_loss / n_batches,
        'aux_loss': total_aux_loss / n_batches,
        'accuracy': correct / max(total, 1),
        'grad_norm_mean': np.mean(grad_norms) if grad_norms else 0,
        'grad_norm_max': np.max(grad_norms) if grad_norms else 0,
        'grad_norm_std': np.std(grad_norms) if grad_norms else 0,
        'nan_batches': nan_batches,
    }


@torch.no_grad()
def validate_multi(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    multi_task: bool = False,
    version: str = 'v2',
) -> dict:
    """Validate multi-condition model (v1, v2, v3)."""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    correct = 0
    total = 0
    valid_batches = 0

    all_preds = []
    all_labels = []
    all_probs = []

    total_kinetic_mse = 0
    total_pattern_mse = 0
    kinetic_samples = 0
    pattern_samples = 0

    for batch in dataloader:
        trajectories = batch['trajectories'].to(device)
        conditions = batch['conditions'].to(device)
        condition_mask = batch['condition_mask'].to(device)
        labels = batch['mechanism_idx'].to(device)

        derived_features = batch.get('derived_features')
        if derived_features is not None:
            derived_features = derived_features.to(device)

        if version == 'v1':
            output = model(trajectories, conditions, condition_mask=condition_mask)
        else:
            output = model(trajectories, conditions, derived_features=derived_features,
                          condition_mask=condition_mask)

        logits = output['logits']
        loss = F.cross_entropy(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        valid_batches += 1
        total_loss += loss.item()
        total_cls_loss += loss.item()

        # Multi-task metrics
        if multi_task and 'kinetic_params' in output and 'kinetic_params' in batch:
            kinetic_params_pred = output['kinetic_params']
            kinetic_params_target = batch['kinetic_params'].to(device)
            valid_mask = ~condition_mask
            kinetic_diff = (kinetic_params_pred - kinetic_params_target) ** 2
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(kinetic_diff)
            n_valid = valid_mask_expanded.float().sum()
            if n_valid > 0:
                kinetic_mse = (kinetic_diff * valid_mask_expanded.float()).sum() / n_valid
                total_kinetic_mse += kinetic_mse.item() * labels.size(0)
                kinetic_samples += labels.size(0)

        if multi_task and 'param_pattern' in output and 'param_pattern' in batch:
            param_pattern_pred = output['param_pattern']
            param_pattern_target = batch['param_pattern'].to(device)
            pattern_mse = F.mse_loss(param_pattern_pred, param_pattern_target)
            total_pattern_mse += pattern_mse.item() * labels.size(0)
            pattern_samples += labels.size(0)

        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for c in range(10):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = float((all_preds[mask] == c).mean())
        else:
            per_class_acc[c] = 0.0

    result = {
        'loss': total_loss / max(valid_batches, 1),
        'cls_loss': total_cls_loss / max(valid_batches, 1),
        'accuracy': correct / max(total, 1),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'per_class_accuracy': per_class_acc,
        'mean_confidence': np.mean([p.max() for p in all_probs]) if all_probs else 0,
    }

    if multi_task:
        result['kinetic_mse'] = total_kinetic_mse / max(kinetic_samples, 1)
        result['pattern_mse'] = total_pattern_mse / max(pattern_samples, 1)

    return result


# =============================================================================
# UTILITIES
# =============================================================================

def print_confusion_matrix(preds, labels, mechanism_names):
    """Print confusion matrix summary."""
    n_classes = len(mechanism_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for p, l in zip(preds, labels):
        confusion[l, p] += 1

    print("\n" + "="*70)
    print("PER-MECHANISM ACCURACY:")
    print("="*70)
    for i, name in enumerate(mechanism_names):
        total = confusion[i].sum()
        correct = confusion[i, i]
        acc = correct / max(total, 1) * 100
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {name:35s}: {acc:5.1f}% |{bar}| ({correct}/{total})")

    print("\n" + "="*70)
    print("MOST CONFUSED PAIRS:")
    print("="*70)
    confused = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and confusion[i, j] > 2:
                confused.append((mechanism_names[i], mechanism_names[j], confusion[i, j]))
    confused.sort(key=lambda x: -x[2])
    for true_m, pred_m, count in confused[:10]:
        print(f"  {true_m:30s} -> {pred_m:30s}: {count}")

    print("\n" + "="*70)
    print("FULL CONFUSION MATRIX:")
    print("="*70)
    header = " " * 25
    for name in mechanism_names:
        header += f"{name[:4]:>5s}"
    print(header)
    for i, name in enumerate(mechanism_names):
        row = f"{name[:25]:25s}"
        for j in range(n_classes):
            if i == j:
                row += f"[{confusion[i,j]:3d}]"
            else:
                row += f" {confusion[i,j]:3d} "
        print(row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TACTIC-Kinetics (all versions)")
    parser.add_argument("--version", type=str, default="v2", choices=["v0", "v1", "v2", "v3"],
                       help="Model version (v0=single-curve, v1=basic-multi, v2=improved, v3=multi-task)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-traj-layers", type=int, default=2, help="Trajectory encoder layers")
    parser.add_argument("--n-cross-layers", type=int, default=3, help="Cross-attention layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--samples-per-mech", type=int, default=1000, help="Samples per mechanism")
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs (e.g., '0,1,2')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory (default: checkpoints/<version>)")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory (default: logs/<version>)")
    parser.add_argument("--log-interval", type=int, default=10, help="Batch logging interval")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval (epochs)")
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset path (default: from version config)")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate dataset")
    parser.add_argument("--n-workers", type=int, default=None, help="CPU workers for data generation")
    parser.add_argument("--aux-weight", type=float, default=0.5, help="Initial auxiliary loss weight (v3 only)")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience (0 = disabled)")
    args = parser.parse_args()

    # Get version config
    version = args.version
    config = VERSION_CONFIGS[version]

    print("="*70)
    print(f"TACTIC-Kinetics Training: {version.upper()}")
    print("="*70)
    print(f"Description: {config['description']}")
    print(f"Expected accuracy: {config['expected_accuracy']}")
    print(f"N conditions: {config['n_conditions']}")
    print(f"N trajectory features: {config['n_traj_features']}")
    print(f"Use derived features: {config['use_derived']}")
    print(f"Use pairwise comparison: {config['use_pairwise']}")
    print(f"Multi-task: {config['multi_task']}")
    print("="*70)

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_ids = None
    device = setup_device(gpu_ids)

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(f"checkpoints/{version}")
    log_dir = Path(args.log_dir) if args.log_dir else Path(f"logs/{version}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(log_dir, log_interval=args.log_interval)

    # Dataset path
    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(config['dataset_path'])

    # Load/generate data based on version
    print("\n" + "="*70)
    print("LOADING/GENERATING DATA")
    print("="*70)

    if version == 'v0':
        # Load v0 single-curve data
        if not dataset_path.exists():
            print(f"ERROR: v0 dataset not found: {dataset_path}")
            print("Generate it first or provide --dataset-path")
            return

        print(f"Loading v0 dataset from {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
        train_samples = data['train_samples']
        val_samples = data['val_samples']

        train_dataset = SingleCurveDataset(train_samples)
        val_dataset = SingleCurveDataset(val_samples)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, collate_fn=single_curve_collate_fn, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=single_curve_collate_fn, pin_memory=True)

    elif version == 'v1':
        # Generate/load v1 basic multi-condition data
        n_conditions = config['n_conditions']

        if dataset_path.exists() and not args.regenerate:
            print(f"Loading v1 dataset from {dataset_path}")
            all_samples, _ = load_dataset(dataset_path)
        else:
            print(f"Generating v1 dataset with {n_conditions} conditions...")
            gen_config = MultiConditionConfig(
                n_conditions_per_sample=n_conditions,
                n_timepoints=20,
                noise_level=0.03,
            )
            generator = MultiConditionGenerator(gen_config, seed=args.seed)
            n_workers = args.n_workers if args.n_workers else cpu_count()
            all_samples = generator.generate_batch(args.samples_per_mech, n_workers=n_workers)
            save_dataset(all_samples, dataset_path, gen_config)

        # Split
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(all_samples))
        rng.shuffle(indices)
        n_val = int(len(all_samples) * 0.2)
        train_samples = [all_samples[i] for i in indices[n_val:]]
        val_samples = [all_samples[i] for i in indices[:n_val]]

        train_dataset = BasicMultiConditionDataset(train_samples, max_conditions=n_conditions)
        val_dataset = BasicMultiConditionDataset(val_samples, max_conditions=n_conditions)
        val_dataset.conc_mean = train_dataset.conc_mean
        val_dataset.conc_std = train_dataset.conc_std
        val_dataset.time_max = train_dataset.time_max

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, collate_fn=basic_multi_condition_collate_fn, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=basic_multi_condition_collate_fn, pin_memory=True)

    else:  # v2 or v3
        n_conditions = config['n_conditions']

        if dataset_path.exists() and not args.regenerate:
            print(f"Loading dataset from {dataset_path}")
            all_samples, _ = load_dataset(dataset_path)
        else:
            print(f"Generating dataset with {n_conditions} conditions...")
            gen_config = MultiConditionConfig(
                n_conditions_per_sample=n_conditions,
                n_timepoints=20,
                noise_level=0.03,
            )
            generator = MultiConditionGenerator(gen_config, seed=args.seed)
            n_workers = args.n_workers if args.n_workers else cpu_count()
            all_samples = generator.generate_batch(args.samples_per_mech, n_workers=n_workers)
            save_dataset(all_samples, dataset_path, gen_config)

        # Split
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(all_samples))
        rng.shuffle(indices)
        n_val = int(len(all_samples) * 0.2)
        train_samples = [all_samples[i] for i in indices[n_val:]]
        val_samples = [all_samples[i] for i in indices[:n_val]]

        dataset_config = MultiConditionDatasetConfig(
            max_conditions=n_conditions,
            n_timepoints=20,
            n_trajectory_features=5,
            n_derived_features=8,
        )

        train_dataset = MultiConditionDataset(train_samples, dataset_config)
        val_dataset = MultiConditionDataset(val_samples, dataset_config)
        val_dataset.conc_mean = train_dataset.conc_mean
        val_dataset.conc_std = train_dataset.conc_std
        val_dataset.time_max = train_dataset.time_max

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, collate_fn=multi_condition_collate_fn, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=multi_condition_collate_fn, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)

    if version == 'v0':
        model = create_single_curve_model(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_traj_layers,
            n_mechanisms=10,
            dropout=args.dropout,
        )
    elif version == 'v1':
        model = create_basic_multi_condition_model(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_traj_layers=args.n_traj_layers,
            n_cross_layers=args.n_cross_layers,
            n_mechanisms=10,
            dropout=args.dropout,
        )
    elif version == 'v2':
        model = create_multi_condition_model(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_traj_layers=args.n_traj_layers,
            n_cross_layers=args.n_cross_layers,
            n_mechanisms=10,
            dropout=args.dropout,
        )
    else:  # v3
        model = create_multi_task_model(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_traj_layers=args.n_traj_layers,
            n_cross_layers=args.n_cross_layers,
            n_mechanisms=10,
            dropout=args.dropout,
        )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model.module if isinstance(model, nn.DataParallel) else model).__name__}")
    print(f"Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Save config
    config_dict = {**vars(args), 'version_config': config}
    with open(log_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    best_val_acc = 0.0
    patience_counter = 0
    multi_task = config['multi_task']

    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        aux_weight = get_aux_weight(epoch, args.epochs, args.aux_weight) if multi_task else 0.0
        if multi_task:
            print(f"  Auxiliary loss weight: {aux_weight:.3f}")

        # Train
        if version == 'v0':
            train_metrics = train_epoch_v0(
                model, train_loader, optimizer, device, epoch, logger,
                args.grad_clip, args.log_interval)
        else:
            train_metrics = train_epoch_multi(
                model, train_loader, optimizer, device, epoch, logger,
                args.grad_clip, args.log_interval, multi_task, aux_weight, version)

        print(f"\nEpoch {epoch+1} Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']*100:.1f}%")

        # Validate
        if (epoch + 1) % args.val_interval == 0:
            if version == 'v0':
                val_metrics = validate_v0(model, val_loader, device)
            else:
                val_metrics = validate_multi(model, val_loader, device, multi_task, version)

            print(f"Epoch {epoch+1} Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']*100:.1f}%")
        else:
            val_metrics = {'loss': 0, 'accuracy': 0, 'predictions': [], 'labels': [], 'per_class_accuracy': {}}

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, aux_weight)
        logger.save_history()

        # Checkpointing
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            print(f"  *** New best: {best_val_acc*100:.1f}% ***")
        else:
            patience_counter += 1

        checkpoint = {
            'epoch': epoch,
            'global_step': logger.global_step,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': logger.history,
            'args': vars(args),
            'version': version,
        }

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0 and val_metrics['predictions']:
            print_confusion_matrix(val_metrics['predictions'], val_metrics['labels'], MECHANISM_NAMES)

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\nEARLY STOPPING at epoch {epoch+1}")
            break

    torch.save(checkpoint, checkpoint_dir / "final_model.pt")
    logger.save_history()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Version: {version}")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Final evaluation
    if version == 'v0':
        val_metrics = validate_v0(model, val_loader, device)
    else:
        val_metrics = validate_multi(model, val_loader, device, multi_task, version)
    print_confusion_matrix(val_metrics['predictions'], val_metrics['labels'], MECHANISM_NAMES)


if __name__ == "__main__":
    main()
