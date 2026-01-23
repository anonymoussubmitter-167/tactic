#!/usr/bin/env python
"""Investigate NaN/Inf batches in training."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tactic_kinetics.models.tactic_model import TACTICKinetics, TACTICLoss
from tactic_kinetics.training.synthetic_data import SyntheticKineticsDataset, kinetics_collate_fn
from torch.utils.data import DataLoader


def check_tensor(name, tensor):
    """Check tensor for NaN/Inf and print stats."""
    if tensor is None:
        print(f"  {name}: None")
        return False

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"  {name}: NaN={has_nan}, Inf={has_inf}, shape={tensor.shape}")
        print(f"    min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            print(f"    NaN count: {nan_count}/{tensor.numel()}")
        return True
    return False


def investigate_batch(model, loss_fn, batch, device):
    """Run forward pass and check for NaN at each stage."""
    print("\n" + "="*60)
    print("Investigating batch...")

    # Move to device
    times = batch["times"].to(device).float()
    values = batch["values"].to(device).float()
    mask = batch["mask"].to(device)
    conditions = batch["conditions"].to(device).float()
    mechanism_idx = batch["mechanism_idx"].to(device)

    print("\n1. Input tensors:")
    has_input_nan = False
    has_input_nan |= check_tensor("times", times)
    has_input_nan |= check_tensor("values", values)
    has_input_nan |= check_tensor("conditions", conditions)
    has_input_nan |= check_tensor("mask", mask)

    if has_input_nan:
        print("  -> NaN in INPUT data!")
        return "input"

    print(f"  times: min={times.min():.4f}, max={times.max():.4f}")
    print(f"  values: min={values.min():.6f}, max={values.max():.6f}")
    print(f"  conditions: min={conditions.min():.4f}, max={conditions.max():.4f}")

    # Check for extreme values
    if values.min() < 1e-10:
        print(f"  -> WARNING: Very small values detected: {values.min():.2e}")
    if values.max() > 1e6:
        print(f"  -> WARNING: Very large values detected: {values.max():.2e}")

    # Forward pass
    print("\n2. Forward pass:")
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(times, values, conditions, mask)
        except Exception as e:
            print(f"  -> EXCEPTION in forward: {e}")
            return "forward_exception"

    has_output_nan = False
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            has_output_nan |= check_tensor(f"output.{key}", val)
        elif isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, torch.Tensor):
                    has_output_nan |= check_tensor(f"output.{key}.{k2}", v2)

    if has_output_nan:
        print("  -> NaN in model OUTPUT!")
        return "output"

    # Loss computation
    print("\n3. Loss computation:")
    targets = {"mechanism_labels": mechanism_idx}
    try:
        losses = loss_fn(outputs, targets)
    except Exception as e:
        print(f"  -> EXCEPTION in loss: {e}")
        return "loss_exception"

    has_loss_nan = False
    for key, val in losses.items():
        has_loss_nan |= check_tensor(f"loss.{key}", val)

    if has_loss_nan:
        print("  -> NaN in LOSS!")
        return "loss"

    print(f"  total loss: {losses['total'].item():.4f}")
    print("  -> Batch OK")
    return "ok"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset_path = Path("data/synthetic_dataset.pt")
    saved_data = torch.load(dataset_path)
    train_samples = saved_data["train_samples"]

    train_dataset = SyntheticKineticsDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,  # Keep order for reproducibility
        collate_fn=kinetics_collate_fn,
        num_workers=0,
    )

    print(f"Dataset size: {len(train_dataset)}")

    # Create model
    print("\nCreating model...")
    model = TACTICKinetics(
        d_model=256,
        n_encoder_heads=8,
        n_encoder_layers=6,
        d_ff=1024,
        n_decoder_layers=[256, 128],
        dropout=0.0,  # Disable dropout for deterministic investigation
        n_conditions=4,
        temperature=298.15,
    ).to(device)

    loss_fn = TACTICLoss(
        lambda_traj=1.0,
        lambda_mech=1.0,
        lambda_thermo=0.1,
        lambda_prior=0.01,
    )

    # Scan batches for NaN
    print("\nScanning batches for NaN...")
    nan_batches = []
    results = {"ok": 0, "input": 0, "output": 0, "loss": 0, "forward_exception": 0, "loss_exception": 0}

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 200:  # Check first 200 batches
            break

        result = investigate_batch(model, loss_fn, batch, device)
        results[result] += 1

        if result != "ok":
            nan_batches.append((batch_idx, result))

            # Deep dive on first NaN batch
            if len(nan_batches) == 1:
                print("\n" + "="*60)
                print("DEEP DIVE on first NaN batch:")
                print(f"Batch {batch_idx}")

                # Check individual samples
                for i in range(min(5, len(batch["times"]))):
                    print(f"\n  Sample {i}:")
                    print(f"    mechanism_idx: {batch['mechanism_idx'][i].item()}")
                    print(f"    values range: [{batch['values'][i].min():.6f}, {batch['values'][i].max():.6f}]")
                    print(f"    times range: [{batch['times'][i].min():.4f}, {batch['times'][i].max():.4f}]")

                    # Check for zeros or extreme values
                    zeros = (batch['values'][i] == 0).sum().item()
                    very_small = (batch['values'][i] < 1e-8).sum().item()
                    print(f"    zeros: {zeros}, very_small (<1e-8): {very_small}")

    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Batches checked: {batch_idx + 1}")
    print(f"  Results: {results}")
    print(f"  NaN batches: {len(nan_batches)}")
    if nan_batches:
        print(f"  First 10 NaN batches: {nan_batches[:10]}")


if __name__ == "__main__":
    main()
