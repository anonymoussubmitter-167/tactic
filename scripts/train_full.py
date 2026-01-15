#!/usr/bin/env python3
"""
Full training script for TACTIC-Kinetics.

This script trains the model on synthetic data from multiple mechanism families.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

from tactic_kinetics.mechanisms.templates import get_all_mechanisms
from tactic_kinetics.training.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    SyntheticKineticsDataset,
    create_dataloaders,
)
from tactic_kinetics.models.encoder import ObservationEncoder
from tactic_kinetics.models.classifier import MechanismClassifier
from tactic_kinetics.models.decoder import EnergyDecoder


class TACTICKineticsLight(nn.Module):
    """
    Lightweight TACTIC model for training without torchdiffeq.

    Focuses on mechanism classification and energy prediction.
    """

    def __init__(
        self,
        mechanism_names: list,
        d_model: int = 256,
        n_encoder_layers: int = 6,
        n_encoder_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_conditions: int = 4,
    ):
        super().__init__()
        self.mechanism_names = mechanism_names
        self.n_mechanisms = len(mechanism_names)
        self.d_model = d_model

        # Encoder
        self.encoder = ObservationEncoder(
            d_model=d_model,
            n_heads=n_encoder_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_observables=1,
            n_conditions=n_conditions,
            condition_names=["temperature", "ph", "S0", "E0"],
        )

        # Classifier
        self.classifier = MechanismClassifier(
            d_input=d_model,
            mechanism_names=mechanism_names,
            hidden_dims=[256, 128],
            dropout=dropout,
        )

        # Energy decoders (one per mechanism)
        # For simplicity, use a shared decoder that outputs max params
        max_state_params = 8  # Maximum across mechanisms
        max_barrier_params = 8

        self.energy_decoder = EnergyDecoder(
            d_input=d_model,
            n_state_energies=max_state_params,
            n_barrier_energies=max_barrier_params,
            hidden_dims=[256, 128],
            dropout=dropout,
        )

    def forward(self, times, values, conditions, mask):
        """Forward pass."""
        # Encode observations
        h = self.encoder(times, values, conditions, mask)

        # Classify mechanism
        mechanism_logits = self.classifier(h)

        # Predict energies
        state_energies, barrier_energies = self.energy_decoder(h)

        return {
            "latent": h,
            "mechanism_logits": mechanism_logits,
            "state_energies": state_energies,
            "barrier_energies": barrier_energies,
        }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        times = batch["times"].to(device)
        values = batch["values"].to(device)
        mask = batch["mask"].to(device)
        conditions = batch["conditions"].to(device)
        labels = batch["mechanism_idx"].to(device)

        optimizer.zero_grad()

        outputs = model(times, values, conditions, mask)

        # Classification loss
        loss = criterion(outputs["mechanism_logits"], labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs["mechanism_logits"].argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")

    return total_loss / len(train_loader), total_correct / total_samples


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for batch in val_loader:
        times = batch["times"].to(device)
        values = batch["values"].to(device)
        mask = batch["mask"].to(device)
        conditions = batch["conditions"].to(device)
        labels = batch["mechanism_idx"].to(device)

        outputs = model(times, values, conditions, mask)

        loss = criterion(outputs["mechanism_logits"], labels)

        total_loss += loss.item()
        preds = outputs["mechanism_logits"].argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(val_loader), total_correct / total_samples, all_preds, all_labels


def compute_confusion_matrix(preds, labels, n_classes):
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def main():
    print("=" * 70)
    print("TACTIC-Kinetics Full Training")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    MECHANISM_NAMES = [
        "michaelis_menten_irreversible",
        "michaelis_menten_reversible",
        "competitive_inhibition",
        "uncompetitive_inhibition",
        "substrate_inhibition",
        "product_inhibition",
    ]

    N_SAMPLES_PER_MECHANISM = 2000
    BATCH_SIZE = 64
    N_EPOCHS = 30
    LEARNING_RATE = 3e-4
    D_MODEL = 256
    N_LAYERS = 6

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    print(f"  Mechanisms: {len(MECHANISM_NAMES)}")
    print(f"  Samples per mechanism: {N_SAMPLES_PER_MECHANISM}")

    config = SyntheticDataConfig(
        mechanism_names=MECHANISM_NAMES,
        n_samples_per_mechanism=N_SAMPLES_PER_MECHANISM,
        n_observations=30,
        noise_std=0.03,
        temperature_range=(288.15, 318.15),  # 15-45°C
        substrate_range=(0.01, 10.0),
    )

    generator = SyntheticDataGenerator(config, seed=42)
    all_samples = generator.generate_dataset()

    # Split data
    np.random.seed(42)
    np.random.shuffle(all_samples)
    n_train = int(len(all_samples) * 0.8)
    n_val = int(len(all_samples) * 0.1)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_dataset = SyntheticKineticsDataset(train_samples, max_obs=50)
    val_dataset = SyntheticKineticsDataset(val_samples, max_obs=50)
    test_dataset = SyntheticKineticsDataset(test_samples, max_obs=50)

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Create model
    print(f"\nCreating model...")
    model = TACTICKineticsLight(
        mechanism_names=MECHANISM_NAMES,
        d_model=D_MODEL,
        n_encoder_layers=N_LAYERS,
        n_encoder_heads=8,
        d_ff=1024,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    print(f"\nTraining for {N_EPOCHS} epochs...")
    print("-" * 70)

    best_val_acc = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    start_time = time.time()

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} ({epoch_time:.1f}s): "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
              f"LR={current_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "mechanism_names": MECHANISM_NAMES,
            }, "checkpoints/best_model.pt")
            print(f"  -> New best model saved! (Val Acc: {val_acc:.4f})")

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {total_time/60:.1f} minutes")

    # Final evaluation on test set
    print(f"\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load("checkpoints/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = compute_confusion_matrix(test_preds, test_labels, len(MECHANISM_NAMES))

    # Print header
    print(f"\n{'':20s}", end="")
    for name in MECHANISM_NAMES:
        print(f"{name[:8]:>10s}", end="")
    print()

    # Print rows
    for i, name in enumerate(MECHANISM_NAMES):
        print(f"{name[:20]:20s}", end="")
        for j in range(len(MECHANISM_NAMES)):
            print(f"{cm[i,j]:10d}", end="")
        print()

    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, name in enumerate(MECHANISM_NAMES):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {name}: {class_acc:.4f} ({class_correct}/{class_total})")

    # Save results
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
        "mechanism_names": MECHANISM_NAMES,
        "history": history,
        "config": {
            "n_samples_per_mechanism": N_SAMPLES_PER_MECHANISM,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
        },
    }

    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to checkpoints/training_results.json")
    print(f"Best model saved to checkpoints/best_model.pt")

    print(f"\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return model, history, results


if __name__ == "__main__":
    main()
