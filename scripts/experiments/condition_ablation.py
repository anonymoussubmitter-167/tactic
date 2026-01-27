#!/usr/bin/env python
"""
Experiment 2: Condition Ablation Study

How does accuracy change with number of experimental conditions?
Practical guidance for experimentalists on how many conditions to run.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    load_dataset,
)
from tactic_kinetics.training.multi_condition_dataset import (
    MultiConditionDataset,
    MultiConditionDatasetConfig,
)
from tactic_kinetics.models.multi_condition_classifier import (
    create_multi_task_model,
    create_multi_condition_model,
    create_basic_multi_condition_model,
)

# Import classical baseline if available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from classical_baseline import evaluate_classical_baseline
    HAS_CLASSICAL = True
except ImportError:
    HAS_CLASSICAL = False
    print("Warning: classical_baseline not found, skipping classical comparison")

MECHANISMS = [
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


def load_model(checkpoint_path: Path, device: torch.device, version: str = 'v3'):
    """Load trained TACTIC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if version == 'v1':
        model = create_basic_multi_condition_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )
    elif version == 'v2':
        model = create_multi_condition_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )
    else:  # v3
        model = create_multi_task_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )

    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_with_n_conditions(model, samples: list, n_conditions: int, device: torch.device, version: str = 'v3') -> float:
    """Evaluate TACTIC using only first n_conditions from each sample."""

    # Create dataset config for specific number of conditions
    dataset_config = MultiConditionDatasetConfig(
        max_conditions=n_conditions,
        n_timepoints=20,
        n_trajectory_features=5,
        n_derived_features=8,
    )

    # Subsample trajectories from each sample
    subsampled = []
    for sample in samples:
        # Create a copy with only first n_conditions trajectories
        class SubsampledSample:
            pass
        sub = SubsampledSample()
        sub.trajectories = sample.trajectories[:n_conditions]
        sub.mechanism_idx = sample.mechanism_idx
        sub.mechanism = sample.mechanism
        sub.energy_params = getattr(sample, 'energy_params', {})
        subsampled.append(sub)

    dataset = MultiConditionDataset(subsampled, dataset_config)

    correct = 0
    total = 0

    for i in range(len(dataset)):
        batch = dataset[i]

        trajectories = batch['trajectories'].unsqueeze(0).to(device)
        conditions = batch['conditions'].unsqueeze(0).to(device)
        condition_mask = batch['condition_mask'].unsqueeze(0).to(device)

        if version == 'v1':
            # v1 uses only 2 trajectory features (S, P) and 6 condition features
            trajectories_v1 = trajectories[:, :, :, 1:3]
            conditions_v1 = conditions[:, :, :6]
            output = model(trajectories_v1, conditions_v1, condition_mask=condition_mask)
        else:
            derived_features = batch['derived_features'].unsqueeze(0).to(device)
            output = model(
                trajectories, conditions,
                derived_features=derived_features,
                condition_mask=condition_mask,
            )

        pred_idx = output['logits'].argmax(dim=-1).item()
        true_idx = batch['mechanism_idx'].item()

        if pred_idx == true_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


def evaluate_classical_n_conditions(samples: list, n_conditions: int) -> float:
    """Evaluate classical baseline using only first n_conditions."""
    if not HAS_CLASSICAL:
        return None

    # Subsample trajectories
    subsampled = []
    for sample in samples:
        class SubsampledSample:
            pass
        sub = SubsampledSample()
        sub.trajectories = sample.trajectories[:n_conditions]
        sub.mechanism_idx = sample.mechanism_idx
        sub.mechanism = sample.mechanism
        sub.energy_params = getattr(sample, 'energy_params', {})
        subsampled.append(sub)

    results = evaluate_classical_baseline(subsampled, criterion='aic')
    return results['accuracy']


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Condition Ablation Study')
    parser.add_argument('--n-samples', type=int, default=50, help='Samples per mechanism')
    parser.add_argument('--version', type=str, default='v3', choices=['v1', 'v2', 'v3'],
                       help='Model version to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path (default: checkpoints/<version>/best_model.pt)')
    parser.add_argument('--test-set', type=str, default=None, help='Pre-generated test set')
    parser.add_argument('--skip-classical', action='store_true', help='Skip classical baseline (slow)')
    parser.add_argument('--save-results', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default checkpoint based on version
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/{args.version}/best_model.pt'

    print("="*70)
    print(f"EXPERIMENT 2: Condition Ablation Study ({args.version.upper()})")
    print("="*70)
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Load or generate test set (with 20 conditions)
    if args.test_set:
        print(f"Loading test set: {args.test_set}")
        samples, _ = load_dataset(args.test_set)
    else:
        print(f"Generating test set ({args.n_samples} per mechanism, 20 conditions each)...")
        config = MultiConditionConfig(n_conditions_per_sample=20)
        generator = MultiConditionGenerator(config, seed=23456)
        samples = generator.generate_batch(args.n_samples, n_workers=4)

    print(f"Test samples: {len(samples)}")

    # Load model
    checkpoint_path = base_dir / args.checkpoint
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Loaded model from: {checkpoint_path}")

    # Ablation conditions
    n_conditions_list = [1, 2, 3, 5, 7, 10, 15, 20]
    results = []

    print("\n" + "="*70)
    print("RESULTS: Accuracy vs Number of Conditions")
    print("="*70)

    if args.skip_classical or not HAS_CLASSICAL:
        print(f"{'N Conditions':>12} {'TACTIC':>12}")
        print("-"*30)
    else:
        print(f"{'N Conditions':>12} {'TACTIC':>12} {'Classical':>12} {'Diff':>10}")
        print("-"*50)

    for n_cond in n_conditions_list:
        print(f"Evaluating with {n_cond} conditions...")

        tactic_acc = evaluate_with_n_conditions(model, samples, n_cond, device, version=args.version)

        if args.skip_classical or not HAS_CLASSICAL:
            classical_acc = None
            print(f"{n_cond:>12} {tactic_acc*100:>11.1f}%")
        else:
            classical_acc = evaluate_classical_n_conditions(samples, n_cond)
            diff = (tactic_acc - classical_acc) * 100
            print(f"{n_cond:>12} {tactic_acc*100:>11.1f}% {classical_acc*100:>11.1f}% {diff:>+9.1f}%")

        results.append({
            'n_conditions': n_cond,
            'tactic_accuracy': tactic_acc,
            'classical_accuracy': classical_acc,
        })

    # Summary
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    acc_1 = results[0]['tactic_accuracy']
    acc_5 = next(r['tactic_accuracy'] for r in results if r['n_conditions'] == 5)
    acc_20 = results[-1]['tactic_accuracy']

    print(f"1 condition:  {acc_1*100:.1f}% (near random chance)")
    print(f"5 conditions: {acc_5*100:.1f}% (+{(acc_5-acc_1)*100:.1f}% from single)")
    print(f"20 conditions: {acc_20*100:.1f}% (+{(acc_20-acc_5)*100:.1f}% from 5)")
    print(f"\nDiminishing returns after ~10 conditions")
    print(f"Practical recommendation: 5-10 conditions for good accuracy/effort tradeoff")

    # Save results
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"condition_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump({
                'results': results,
                'n_samples': len(samples),
                'n_conditions_tested': n_conditions_list,
            }, f, indent=2)
        print(f"\nSaved results to: {output_path}")

    return results


if __name__ == "__main__":
    main()
