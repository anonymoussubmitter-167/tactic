"""
Training loop and utilities for TACTIC-Kinetics.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..models.tactic_model import TACTICKinetics, TACTICLoss


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    n_epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Loss weights
    lambda_traj: float = 1.0
    lambda_mech: float = 1.0
    lambda_thermo: float = 0.1
    lambda_prior: float = 0.01

    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "onecycle"

    # Logging
    log_interval: int = 100
    val_interval: int = 1  # Validate every N epochs
    save_interval: int = 10  # Save checkpoint every N epochs

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Mixed precision
    use_amp: bool = True


class TACTICTrainer:
    """
    Trainer for TACTIC-Kinetics model.
    """

    def __init__(
        self,
        model: TACTICKinetics,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # Loss function
        self.loss_fn = TACTICLoss(
            lambda_traj=config.lambda_traj,
            lambda_mech=config.lambda_mech,
            lambda_thermo=config.lambda_thermo,
            lambda_prior=config.lambda_prior,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * config.n_epochs
        warmup_steps = len(train_loader) * config.warmup_epochs

        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.n_epochs,
                eta_min=config.learning_rate * 0.01,
            )
        elif config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
            )
        else:
            self.scheduler = None

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "mechanism_accuracy": [],
            "learning_rate": [],
        }

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mech_correct = 0
        total_samples = 0
        loss_components = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            times = batch["times"].to(self.device)
            values = batch["values"].to(self.device)
            mask = batch["mask"].to(self.device)
            conditions = batch["conditions"].to(self.device)
            mechanism_idx = batch["mechanism_idx"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(times, values, conditions, mask)
                    targets = {"mechanism_labels": mechanism_idx}
                    losses = self.loss_fn(outputs, targets)
                    loss = losses["total"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(times, values, conditions, mask)
                targets = {"mechanism_labels": mechanism_idx}
                losses = self.loss_fn(outputs, targets)
                loss = losses["total"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.optimizer.step()

            # Update scheduler (if per-step)
            if self.config.scheduler == "onecycle":
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            for key, val in losses.items():
                if key != "total":
                    loss_components[key] = loss_components.get(key, 0) + val.item()

            # Mechanism accuracy
            if "mechanism_logits" in outputs:
                preds = outputs["mechanism_logits"].argmax(dim=-1)
                total_mech_correct += (preds == mechanism_idx).sum().item()
            total_samples += times.shape[0]

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": total_mech_correct / total_samples if total_samples > 0 else 0,
            })

        # Compute averages
        n_batches = len(self.train_loader)
        metrics = {
            "loss": total_loss / n_batches,
            "mechanism_accuracy": total_mech_correct / total_samples,
        }
        for key, val in loss_components.items():
            metrics[f"loss_{key}"] = val / n_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_mech_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            times = batch["times"].to(self.device)
            values = batch["values"].to(self.device)
            mask = batch["mask"].to(self.device)
            conditions = batch["conditions"].to(self.device)
            mechanism_idx = batch["mechanism_idx"].to(self.device)

            outputs = self.model(times, values, conditions, mask)
            targets = {"mechanism_labels": mechanism_idx}
            losses = self.loss_fn(outputs, targets)

            total_loss += losses["total"].item()

            if "mechanism_logits" in outputs:
                preds = outputs["mechanism_logits"].argmax(dim=-1)
                total_mech_correct += (preds == mechanism_idx).sum().item()
            total_samples += times.shape[0]

        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "val_mechanism_accuracy": total_mech_correct / total_samples,
        }

        return metrics

    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{self.epoch}.pt"

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "history": self.history,
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

    def train(self, n_epochs: Optional[int] = None):
        """
        Main training loop.

        Args:
            n_epochs: Number of epochs to train (overrides config)
        """
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        print(f"Training for {n_epochs} epochs on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(n_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["mechanism_accuracy"].append(train_metrics["mechanism_accuracy"])

            # Update scheduler (if per-epoch)
            if self.config.scheduler == "cosine" and self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            # Validate
            if self.val_loader and (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics["val_loss"])

                # Check for best model
                is_best = val_metrics["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val_loss"]

                print(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Acc={train_metrics['mechanism_accuracy']:.4f}, "
                    f"Val Loss={val_metrics['val_loss']:.4f}, "
                    f"Val Acc={val_metrics['val_mechanism_accuracy']:.4f}, "
                    f"LR={current_lr:.2e}"
                )
            else:
                is_best = False
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Acc={train_metrics['mechanism_accuracy']:.4f}, "
                    f"LR={current_lr:.2e}"
                )

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(is_best=is_best)

        # Final save
        self.save_checkpoint()

        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed/60:.2f} minutes")

        # Save training history
        history_path = Path(self.config.log_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history


def train_tactic_model(
    mechanism_names: Optional[List[str]] = None,
    n_samples_per_mechanism: int = 1000,
    config: Optional[TrainingConfig] = None,
    seed: int = 42,
) -> TACTICKinetics:
    """
    Convenience function to train a TACTIC model from scratch.

    Args:
        mechanism_names: List of mechanisms to train on
        n_samples_per_mechanism: Number of synthetic samples per mechanism
        config: Training configuration
        seed: Random seed

    Returns:
        Trained model
    """
    from .synthetic_data import generate_mechanism_dataset, create_dataloaders

    # Set seed
    torch.manual_seed(seed)

    # Generate data
    print("Generating synthetic data...")
    train_dataset, val_dataset = generate_mechanism_dataset(
        mechanism_names=mechanism_names,
        n_samples_per_mechanism=n_samples_per_mechanism,
        seed=seed,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create config
    if config is None:
        config = TrainingConfig()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config.batch_size,
    )

    # Create model
    model = TACTICKinetics(
        mechanism_names=mechanism_names,
        d_model=256,
        n_encoder_layers=6,
    )

    # Create trainer and train
    trainer = TACTICTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.train()

    return model
