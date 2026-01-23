#!/usr/bin/env python
"""
Test script to verify each stage of the TACTIC-Kinetics training pipeline.
Run with: python test_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import torch
import torch.nn as nn

def test_stage_1_config():
    """Stage 1: Config Loading"""
    print("=" * 60)
    print("Stage 1: Config Loading")
    print("=" * 60)

    with open("configs/quick_test.yaml") as f:
        config = yaml.safe_load(f)

    print(f"Model config: {config['model']}")
    print(f"Training config: {config['training']}")
    print(f"Learning rate type: {type(config['training']['learning_rate'])}")
    print(f"Learning rate value: {config['training']['learning_rate']}")

    assert isinstance(config['training']['learning_rate'], float), "Learning rate should be float!"
    print("✓ Stage 1 PASSED\n")
    return config


def test_stage_2_gpu():
    """Stage 2: GPU Setup"""
    print("=" * 60)
    print("Stage 2: GPU Setup")
    print("=" * 60)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda:0")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    print(f"Primary device: {device}")
    print("✓ Stage 2 PASSED\n")
    return device


def test_stage_3_model(config, device):
    """Stage 3: Model Creation"""
    print("=" * 60)
    print("Stage 3: Model Creation")
    print("=" * 60)

    from tactic_kinetics.models.tactic_model import TACTICKinetics

    model_config = config.get("model", {})
    model = TACTICKinetics(
        d_model=model_config.get("d_model", 256),
        n_encoder_heads=model_config.get("n_encoder_heads", 8),
        n_encoder_layers=model_config.get("n_encoder_layers", 6),
        d_ff=model_config.get("d_ff", 1024),
        n_decoder_layers=model_config.get("n_decoder_layers", [256, 128]),
        dropout=model_config.get("dropout", 0.1),
        n_conditions=model_config.get("n_conditions", 4),
        temperature=model_config.get("temperature", 298.15),
        use_adjoint=model_config.get("use_adjoint", False),
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Model device: {next(model.parameters()).device}")
    print("✓ Stage 3 PASSED\n")
    return model


def test_stage_4_data():
    """Stage 4: Data Generation"""
    print("=" * 60)
    print("Stage 4: Data Generation (small batch)")
    print("=" * 60)

    from tactic_kinetics.training.synthetic_data import (
        SyntheticDataGenerator, SyntheticDataConfig
    )

    config = SyntheticDataConfig(
        n_samples_per_mechanism=10,  # Small for testing
        n_timepoints=30,
        t_max=100.0,
    )
    generator = SyntheticDataGenerator(config, seed=42)
    print(f"Mechanisms: {generator.mechanism_names}")

    samples = generator.generate_dataset(n_samples_per_mechanism=10)
    print(f"Total samples: {len(samples)}")
    print(f"Sample keys: {list(samples[0].keys())}")
    print(f"State energies shape: {samples[0]['state_energies'].shape}")
    print(f"Barrier energies shape: {samples[0]['barrier_energies'].shape}")
    print(f"Values shape: {samples[0]['values'].shape}")
    print("✓ Stage 4 PASSED\n")
    return samples


def test_stage_5_dataloader():
    """Stage 5: DataLoader Creation"""
    print("=" * 60)
    print("Stage 5: DataLoader Creation")
    print("=" * 60)

    from tactic_kinetics.training.synthetic_data import (
        SyntheticDataConfig, generate_mechanism_dataset, create_dataloaders
    )

    config = SyntheticDataConfig(
        n_samples_per_mechanism=20,
        n_timepoints=30,
        t_max=100.0,
    )

    train_dataset, val_dataset = generate_mechanism_dataset(config=config, seed=42)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=8, num_workers=0
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Get one batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch values shape: {batch['values'].shape}")
    print(f"State energies shape: {batch['state_energies'].shape} (padded)")
    print(f"Barrier energies shape: {batch['barrier_energies'].shape} (padded)")
    print("✓ Stage 5 PASSED\n")
    return train_loader, val_loader


def test_stage_6_training_config(config, device):
    """Stage 6: TrainingConfig Creation"""
    print("=" * 60)
    print("Stage 6: TrainingConfig Creation")
    print("=" * 60)

    from tactic_kinetics.training.trainer import TrainingConfig

    training = config.get("training", {})
    loss = config.get("loss", {})
    trajectory = config.get("trajectory", {})

    train_config = TrainingConfig(
        learning_rate=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        batch_size=int(training.get("batch_size", 32)),
        n_epochs=int(training.get("n_epochs", 100)),
        warmup_epochs=int(training.get("warmup_epochs", 5)),
        grad_clip=float(training.get("grad_clip", 1.0)),
        scheduler=str(training.get("scheduler", "cosine")),
        lambda_traj=float(loss.get("lambda_traj", 1.0)),
        lambda_mech=float(loss.get("lambda_mech", 1.0)),
        lambda_thermo=float(loss.get("lambda_thermo", 0.1)),
        lambda_prior=float(loss.get("lambda_prior", 0.01)),
        use_trajectory_loss=bool(trajectory.get("use_trajectory_loss", True)),
        device=str(device),
    )

    print(f"Learning rate: {train_config.learning_rate}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"N epochs: {train_config.n_epochs}")
    print(f"Device: {train_config.device}")
    print("✓ Stage 6 PASSED\n")
    return train_config


def test_stage_7_trainer(model, train_config, train_loader, val_loader):
    """Stage 7: Trainer Initialization"""
    print("=" * 60)
    print("Stage 7: Trainer Initialization")
    print("=" * 60)

    from tactic_kinetics.training.trainer import TACTICTrainer

    # Handle DataParallel wrapped model
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    trainer = TACTICTrainer(
        model=base_model,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print(f"Optimizer: {type(trainer.optimizer).__name__}")
    print(f"Scheduler: {type(trainer.scheduler).__name__}")
    print(f"Loss function: {type(trainer.loss_fn).__name__}")
    print("✓ Stage 7 PASSED\n")
    return trainer


def test_stage_8_forward_pass(model, train_loader, device):
    """Stage 8: Forward Pass"""
    print("=" * 60)
    print("Stage 8: Forward Pass")
    print("=" * 60)

    model.eval()
    batch = next(iter(train_loader))

    # Move batch to device (ensure float32)
    values = batch["values"].to(device).float()
    times = batch["times"].to(device).float()
    mask = batch["mask"].to(device)
    conditions = batch["conditions"].to(device).float()

    print(f"Input values shape: {values.shape}, dtype: {values.dtype}")
    print(f"Input times shape: {times.shape}, dtype: {times.dtype}")
    print(f"Conditions dtype: {conditions.dtype}")

    with torch.no_grad():
        # Model signature: forward(times, values, conditions, mask)
        outputs = model(times, values, conditions, mask)

    print(f"Output keys: {list(outputs.keys())}")
    print(f"Mechanism logits shape: {outputs['mechanism_logits'].shape}")
    print(f"Energies dict keys: {list(outputs['energies'].keys())}")
    # Show one mechanism's energy shapes
    first_mech = list(outputs['energies'].keys())[0]
    state_e, barrier_e = outputs['energies'][first_mech]
    print(f"  {first_mech} state energies: {state_e.shape}, barrier energies: {barrier_e.shape}")
    print("✓ Stage 8 PASSED\n")
    return outputs


def test_stage_9_loss_computation(trainer, train_loader, device):
    """Stage 9: Loss Computation"""
    print("=" * 60)
    print("Stage 9: Loss Computation")
    print("=" * 60)

    batch = next(iter(train_loader))

    # Move batch to device (ensure float32)
    values = batch["values"].to(device).float()
    times = batch["times"].to(device).float()
    mask = batch["mask"].to(device)
    conditions = batch["conditions"].to(device).float()
    mechanism_idx = batch["mechanism_idx"].to(device)
    state_energies = batch["state_energies"].to(device).float()
    barrier_energies = batch["barrier_energies"].to(device).float()

    trainer.model.eval()
    with torch.no_grad():
        # Model signature: forward(times, values, conditions, mask)
        outputs = trainer.model(times, values, conditions, mask)

    # Compute mechanism loss
    mech_loss = nn.CrossEntropyLoss()(outputs["mechanism_logits"], mechanism_idx)
    print(f"Mechanism classification loss: {mech_loss.item():.4f}")

    # Energy prediction loss - compare against one mechanism's predictions
    # In practice, training uses the TACTICLoss which handles this properly
    if outputs["energies"]:
        first_mech = list(outputs["energies"].keys())[0]
        pred_state, pred_barrier = outputs["energies"][first_mech]
        print(f"Sample energy predictions for {first_mech}:")
        print(f"  State energies: {pred_state[0].detach().cpu().numpy()}")
        print(f"  Barrier energies: {pred_barrier[0].detach().cpu().numpy()}")

    print("✓ Stage 9 PASSED\n")


def test_stage_10_backward_pass(trainer, train_loader, device):
    """Stage 10: Backward Pass"""
    print("=" * 60)
    print("Stage 10: Backward Pass (one step)")
    print("=" * 60)

    trainer.model.train()
    trainer.optimizer.zero_grad()

    batch = next(iter(train_loader))

    # Move batch to device (ensure float32)
    values = batch["values"].to(device).float()
    times = batch["times"].to(device).float()
    mask = batch["mask"].to(device)
    conditions = batch["conditions"].to(device).float()
    mechanism_idx = batch["mechanism_idx"].to(device)
    state_energies = batch["state_energies"].to(device).float()
    barrier_energies = batch["barrier_energies"].to(device).float()

    # Forward pass - Model signature: forward(times, values, conditions, mask)
    outputs = trainer.model(times, values, conditions, mask)

    # Compute losses using the trainer's loss function
    targets = {"mechanism_labels": mechanism_idx}
    losses = trainer.loss_fn(outputs, targets)

    total_loss = losses["total"]
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items() if k != 'total')}")

    # Backward pass
    total_loss.backward()

    # Check gradients
    total_grad_norm = 0.0
    for p in trainer.model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Gradient norm: {total_grad_norm:.4f}")

    # Optimizer step
    trainer.optimizer.step()
    print("Optimizer step completed")

    print("✓ Stage 10 PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("TACTIC-Kinetics Pipeline Test")
    print("=" * 60 + "\n")

    try:
        # Stage 1: Config
        config = test_stage_1_config()

        # Stage 2: GPU
        device = test_stage_2_gpu()

        # Stage 3: Model
        model = test_stage_3_model(config, device)

        # Stage 4: Data Generation
        test_stage_4_data()

        # Stage 5: DataLoader
        train_loader, val_loader = test_stage_5_dataloader()

        # Stage 6: TrainingConfig
        train_config = test_stage_6_training_config(config, device)

        # Stage 7: Trainer
        trainer = test_stage_7_trainer(model, train_config, train_loader, val_loader)

        # Stage 8: Forward Pass
        test_stage_8_forward_pass(model, train_loader, device)

        # Stage 9: Loss Computation
        test_stage_9_loss_computation(trainer, train_loader, device)

        # Stage 10: Backward Pass
        test_stage_10_backward_pass(trainer, train_loader, device)

        print("=" * 60)
        print("ALL STAGES PASSED!")
        print("=" * 60)
        print("\nYou can now run full training with:")
        print("  python train.py --config configs/quick_test.yaml --gpus 4,5,6")
        print("Or:")
        print("  python train.py --config configs/default.yaml --gpus 4,5,6")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
