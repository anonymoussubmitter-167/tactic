#!/usr/bin/env python
"""
Generate and save synthetic dataset for TACTIC-Kinetics training.

Run once to generate data, then train.py will load from disk.

Usage:
    python generate_dataset.py --config configs/default.yaml
    python generate_dataset.py --samples 5000  # 5000 per mechanism
"""

import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from tactic_kinetics.training.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    SyntheticKineticsDataset,
)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Samples per mechanism (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_dataset.pt",
        help="Output path for dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        data_config = config.get("data", {})
        n_samples = data_config.get("n_samples_per_mechanism", args.samples)
        n_timepoints = data_config.get("n_timepoints", 50)
        t_max = data_config.get("t_max", 100.0)
        noise_std = data_config.get("noise_std", 0.02)
    else:
        n_samples = args.samples
        n_timepoints = 50
        t_max = 100.0
        noise_std = 0.02

    print(f"Generating synthetic dataset:")
    print(f"  Samples per mechanism: {n_samples}")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  t_max: {t_max}")
    print(f"  Noise std: {noise_std}")
    print(f"  Seed: {args.seed}")
    print()

    # Create config
    syn_config = SyntheticDataConfig(
        n_samples_per_mechanism=n_samples,
        n_timepoints=n_timepoints,
        t_max=t_max,
        noise_std=noise_std,
        use_thermodynamic_priors=True,
    )

    # Generate data
    generator = SyntheticDataGenerator(syn_config, seed=args.seed)
    samples = generator.generate_dataset()

    print(f"\nTotal samples generated: {len(samples)}")

    # Split 80/20
    import numpy as np
    np.random.seed(args.seed)
    np.random.shuffle(samples)

    n_train = int(len(samples) * 0.8)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "train_samples": train_samples,
        "val_samples": val_samples,
        "config": {
            "n_samples_per_mechanism": n_samples,
            "n_timepoints": n_timepoints,
            "t_max": t_max,
            "noise_std": noise_std,
            "seed": args.seed,
        },
        "mechanism_names": generator.mechanism_names,
    }, output_path)

    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
