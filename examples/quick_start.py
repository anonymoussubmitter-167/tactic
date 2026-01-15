#!/usr/bin/env python3
"""
TACTIC-Kinetics Quick Start Example

This script demonstrates the basic usage of TACTIC-Kinetics:
1. Creating mechanism templates
2. Generating synthetic data
3. Building and training the model
4. Inferring mechanism and energy landscape
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '..')

from tactic_kinetics import (
    TACTICKinetics,
    TACTICLoss,
    get_all_mechanisms,
    get_mechanism_by_name,
)
from tactic_kinetics.training.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    generate_mechanism_dataset,
    create_dataloaders,
)
from tactic_kinetics.training.trainer import TACTICTrainer, TrainingConfig
from tactic_kinetics.utils.thermodynamics import (
    eyring_rate_constant,
    gibbs_to_keq,
)


def demo_mechanisms():
    """Demonstrate mechanism templates."""
    print("=" * 60)
    print("MECHANISM TEMPLATES")
    print("=" * 60)

    mechanisms = get_all_mechanisms()
    print(f"\nAvailable mechanisms ({len(mechanisms)}):")
    for name, mech in mechanisms.items():
        print(f"  - {name}: {mech.n_states} states, {mech.n_transitions} transitions, {mech.n_total_params} params")

    # Get details for Michaelis-Menten
    mm = get_mechanism_by_name("michaelis_menten_irreversible")
    print(f"\n{mm.describe()}")


def demo_thermodynamics():
    """Demonstrate thermodynamic utilities."""
    print("\n" + "=" * 60)
    print("THERMODYNAMIC CALCULATIONS")
    print("=" * 60)

    # Eyring equation example
    temperature = 298.15  # K
    activation_energy = 60.0  # kJ/mol

    rate_constant = eyring_rate_constant(activation_energy, temperature)
    print(f"\nEyring equation:")
    print(f"  Activation energy: {activation_energy} kJ/mol")
    print(f"  Temperature: {temperature} K")
    print(f"  Rate constant: {rate_constant:.2e} s^-1")

    # Equilibrium constant
    dg_rxn = -10.0  # kJ/mol (favorable reaction)
    keq = gibbs_to_keq(dg_rxn, temperature)
    print(f"\nEquilibrium constant:")
    print(f"  ΔG° = {dg_rxn} kJ/mol")
    print(f"  K_eq = {keq:.2f}")


def demo_synthetic_data():
    """Demonstrate synthetic data generation."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Configure data generation
    config = SyntheticDataConfig(
        mechanism_names=["michaelis_menten_irreversible", "competitive_inhibition"],
        n_samples_per_mechanism=100,
        n_observations=20,
        noise_std=0.02,
    )

    generator = SyntheticDataGenerator(config, seed=42)

    # Generate a single sample
    sample = generator.generate_sample("michaelis_menten_irreversible")

    print(f"\nGenerated sample:")
    print(f"  Mechanism: {sample['mechanism_name']}")
    print(f"  State energies: {sample['state_energies'].numpy()}")
    print(f"  Barrier energies: {sample['barrier_energies'].numpy()}")
    print(f"  Conditions: {sample['conditions'].numpy()}")
    print(f"  Number of observations: {len(sample['times'])}")

    return generator


def demo_model():
    """Demonstrate model creation and inference."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    # Create model for subset of mechanisms
    mechanism_names = ["michaelis_menten_irreversible", "competitive_inhibition"]
    model = TACTICKinetics(
        mechanism_names=mechanism_names,
        d_model=128,  # Smaller for demo
        n_encoder_layers=3,
    )

    print(f"\nModel created:")
    print(f"  Mechanisms: {mechanism_names}")
    print(f"  Encoder dimension: {model.d_model}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    batch_size = 4
    n_obs = 20

    times = torch.linspace(0, 100, n_obs).unsqueeze(0).expand(batch_size, -1)
    values = torch.randn(batch_size, n_obs).abs()  # Dummy values
    conditions = torch.randn(batch_size, 4)  # T, pH, S0, E0
    mask = torch.ones(batch_size, n_obs, dtype=torch.bool)

    # Forward pass
    with torch.no_grad():
        outputs = model(times, values, conditions, mask)

    print(f"\nForward pass:")
    print(f"  Latent shape: {outputs['latent'].shape}")
    print(f"  Mechanism probs: {outputs['mechanism_probs'][0].numpy()}")
    print(f"  Predicted mechanism: {mechanism_names[outputs['mechanism_probs'][0].argmax()]}")

    return model


def demo_training(quick=True):
    """Demonstrate training pipeline."""
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    # Use minimal settings for demo
    mechanism_names = ["michaelis_menten_irreversible", "competitive_inhibition"]

    # Generate small dataset
    train_dataset, val_dataset = generate_mechanism_dataset(
        mechanism_names=mechanism_names,
        n_samples_per_mechanism=50 if quick else 500,
        seed=42,
    )

    print(f"\nDataset size:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=16,
        num_workers=0,
    )

    # Create model
    model = TACTICKinetics(
        mechanism_names=mechanism_names,
        d_model=64,  # Very small for demo
        n_encoder_layers=2,
    )

    # Create trainer
    config = TrainingConfig(
        n_epochs=3 if quick else 20,
        learning_rate=1e-3,
        batch_size=16,
        device="cpu",  # Use CPU for demo
        use_amp=False,
        checkpoint_dir="checkpoints_demo",
        log_dir="logs_demo",
    )

    trainer = TACTICTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print(f"\nTraining for {config.n_epochs} epochs...")
    history = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final mechanism accuracy: {history['mechanism_accuracy'][-1]:.4f}")

    return model, history


def main():
    """Run all demos."""
    print("TACTIC-Kinetics Quick Start Demo")
    print("=" * 60)

    # Run demos
    demo_mechanisms()
    demo_thermodynamics()
    demo_synthetic_data()
    model = demo_model()

    # Training demo (optional, slow)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training demo")
    args = parser.parse_args()

    if args.train:
        demo_training(quick=True)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
