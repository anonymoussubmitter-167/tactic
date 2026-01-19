#!/usr/bin/env python
"""
Multi-condition training script for TACTIC-Kinetics.

This script trains the MultiConditionClassifier on samples where each
sample contains multiple trajectories from the same enzyme under
different experimental conditions.

Usage:
    python train.py --epochs 100 --batch-size 32
    python train.py --gpus 0,1,2,3 --epochs 200
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))

from tactic_kinetics.models.multi_condition_classifier import (
    MultiConditionClassifier,
    create_multi_condition_model,
)
from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    generate_and_save_dataset,
    load_dataset,
    save_dataset,
)
from tactic_kinetics.training.multi_condition_dataset import (
    MultiConditionDataset,
    MultiConditionDatasetConfig,
    multi_condition_collate_fn,
)
from multiprocessing import cpu_count


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

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        """Log epoch-level metrics."""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['learning_rate'].append(lr)
        self.history['epoch_grad_norm_mean'].append(train_metrics.get('grad_norm_mean', 0))
        self.history['epoch_grad_norm_max'].append(train_metrics.get('grad_norm_max', 0))

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger,
    grad_clip: float = 1.0,
    log_interval: int = 10,
) -> dict:
    """Train for one epoch with detailed logging."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    nan_batches = 0

    # Track gradient norms
    grad_norms = []

    for batch_idx, batch in enumerate(dataloader):
        trajectories = batch['trajectories'].to(device)
        conditions = batch['conditions'].to(device)
        derived_features = batch['derived_features'].to(device)
        condition_mask = batch['condition_mask'].to(device)
        labels = batch['mechanism_idx'].to(device)
        batch_size = labels.size(0)

        optimizer.zero_grad()

        output = model(trajectories, conditions, derived_features=derived_features, condition_mask=condition_mask)
        logits = output['logits']

        loss = F.cross_entropy(logits, labels)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
            continue

        loss.backward()

        # Compute gradient norm before clipping
        grad_norm = compute_grad_norm(model)
        grad_norms.append(grad_norm)

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        batch_acc = batch_correct / batch_size

        total_loss += loss.item()
        correct += batch_correct
        total += batch_size

        # Log every log_interval batches
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
                  f"GradNorm: {grad_norm:.2f} | "
                  f"LR: {current_lr:.2e}")

    # Epoch statistics
    avg_loss = total_loss / max(len(dataloader) - nan_batches, 1)
    accuracy = correct / max(total, 1)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'grad_norm_mean': np.mean(grad_norms) if grad_norms else 0,
        'grad_norm_max': np.max(grad_norms) if grad_norms else 0,
        'grad_norm_std': np.std(grad_norms) if grad_norms else 0,
        'nan_batches': nan_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        trajectories = batch['trajectories'].to(device)
        conditions = batch['conditions'].to(device)
        derived_features = batch['derived_features'].to(device)
        condition_mask = batch['condition_mask'].to(device)
        labels = batch['mechanism_idx'].to(device)

        output = model(trajectories, conditions, derived_features=derived_features, condition_mask=condition_mask)
        logits = output['logits']

        loss = F.cross_entropy(logits, labels)

        # Skip NaN losses
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

    # Compute per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for c in range(10):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == c).mean()
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

    # Most confused pairs
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

    # Print full matrix
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


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Condition TACTIC-Kinetics")
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
    parser.add_argument("--n-conditions", type=int, default=20, help="Conditions per sample")
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs (e.g., '0,1,2')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Batch logging interval")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval (epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--dataset-path", type=str, default="data/multi_condition_dataset.pt", help="Path to save/load dataset")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate dataset even if exists")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of CPU workers for data generation (default: all cores)")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_ids = None
    device = setup_device(gpu_ids)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(log_dir, log_interval=args.log_interval)

    # Generate or load data
    print("\n" + "="*70)
    print("LOADING/GENERATING MULTI-CONDITION DATA")
    print("="*70)

    n_workers = args.n_workers if args.n_workers else cpu_count()
    print(f"CPU cores available: {cpu_count()}, using: {n_workers}")

    dataset_path = Path(args.dataset_path)

    if dataset_path.exists() and not args.regenerate:
        print(f"\nLoading existing dataset from {dataset_path}")
        all_samples, saved_config = load_dataset(dataset_path)
        print(f"Loaded {len(all_samples)} samples")

        # Use saved config's mechanism list
        gen_config = saved_config if saved_config else MultiConditionConfig(
            n_conditions_per_sample=args.n_conditions,
        )
        generator = MultiConditionGenerator(gen_config, seed=args.seed)
    else:
        print(f"\nGenerating {args.samples_per_mech} samples per mechanism using {n_workers} workers...")

        gen_config = MultiConditionConfig(
            n_conditions_per_sample=args.n_conditions,
            n_timepoints=20,
            noise_level=0.03,
        )
        generator = MultiConditionGenerator(gen_config, seed=args.seed)
        all_samples = generator.generate_batch(args.samples_per_mech, n_workers=n_workers)

        # Save for future runs
        print(f"\nSaving dataset to {dataset_path}")
        save_dataset(all_samples, dataset_path, gen_config)

    print(f"Total samples: {len(all_samples)}")

    # Split train/val
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)

    n_val = int(len(all_samples) * 0.2)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Create datasets
    dataset_config = MultiConditionDatasetConfig(
        max_conditions=args.n_conditions,  # 20 conditions max
        n_timepoints=20,
        n_trajectory_features=5,  # t, S, P, dS/dt, dP/dt
        n_derived_features=8,  # v0, t_half, etc.
    )

    train_dataset = MultiConditionDataset(train_samples, dataset_config)
    val_dataset = MultiConditionDataset(val_samples, dataset_config)

    # Share normalization
    val_dataset.conc_mean = train_dataset.conc_mean
    val_dataset.conc_std = train_dataset.conc_std
    val_dataset.time_max = train_dataset.time_max

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=multi_condition_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=multi_condition_collate_fn,
        pin_memory=True,
    )

    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)

    model = create_multi_condition_model(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_traj_layers=args.n_traj_layers,
        n_cross_layers=args.n_cross_layers,
        n_mechanisms=10,
        dropout=args.dropout,
        n_condition_features=8,
        n_traj_features=5,  # t, S, P, dS/dt, dP/dt
        n_derived_features=8,  # v0, t_half, rate_ratio, etc.
    )
    model = model.to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Resume if specified
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.history = checkpoint.get('history', logger.history)
        logger.global_step = checkpoint.get('global_step', 0)

    # Save config
    config_path = log_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Log interval: {args.log_interval} batches, Val interval: {args.val_interval} epochs")
    print("="*70)

    mechanism_names = generator.MECHANISMS

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, logger, args.grad_clip, args.log_interval
        )

        print(f"\nEpoch {epoch+1} Train Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']*100:.1f}%")
        print(f"  Grad Norm: mean={train_metrics['grad_norm_mean']:.2f}, "
              f"max={train_metrics['grad_norm_max']:.2f}, "
              f"std={train_metrics['grad_norm_std']:.2f}")
        if train_metrics['nan_batches'] > 0:
            print(f"  WARNING: {train_metrics['nan_batches']} NaN batches skipped")

        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device)

            print(f"\nEpoch {epoch+1} Validation Summary:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']*100:.1f}%")
            print(f"  Mean Confidence: {val_metrics['mean_confidence']:.3f}")
        else:
            val_metrics = {'loss': 0, 'accuracy': 0, 'predictions': [], 'labels': []}

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Learning Rate: {current_lr:.2e}")

        # Log epoch
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
        logger.save_history()

        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            print(f"\n  *** New best validation accuracy: {best_val_acc*100:.1f}% ***")

        checkpoint = {
            'epoch': epoch,
            'global_step': logger.global_step,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': logger.history,
            'args': vars(args),
        }

        # Save every 10 epochs and best
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")

        # Print confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0 and val_metrics['predictions']:
            print_confusion_matrix(val_metrics['predictions'], val_metrics['labels'], mechanism_names)

    # Final save
    torch.save(checkpoint, checkpoint_dir / "final_model.pt")
    logger.save_history()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")

    # Final confusion matrix
    val_metrics = validate(model, val_loader, device)
    print_confusion_matrix(val_metrics['predictions'], val_metrics['labels'], mechanism_names)


if __name__ == "__main__":
    main()
