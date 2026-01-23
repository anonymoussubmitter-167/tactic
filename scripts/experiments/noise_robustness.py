#!/usr/bin/env python
"""
Experiment 3: Noise Robustness Analysis

How do TACTIC and classical methods degrade with increasing measurement noise?
Shows robustness to real-world experimental error.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from copy import deepcopy

from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    load_dataset,
)
from tactic_kinetics.training.multi_condition_dataset import MultiConditionDataset
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


def add_noise_to_samples(samples: list, noise_level: float, seed: int = None) -> list:
    """Add Gaussian noise to trajectory data."""
    if seed is not None:
        np.random.seed(seed)

    noisy_samples = []
    for sample in samples:
        # Deep copy the sample
        class NoisySample:
            pass
        noisy = NoisySample()
        noisy.mechanism_idx = sample.mechanism_idx
        noisy.mechanism = sample.mechanism
        noisy.energy_params = getattr(sample, 'energy_params', {})
        noisy.trajectories = []

        for traj in sample.trajectories:
            noisy_traj = {
                't': traj['t'].copy(),
                'conditions': traj['conditions'].copy(),
                'concentrations': {},
            }
            for species, conc in traj['concentrations'].items():
                # Add relative Gaussian noise
                conc_array = np.array(conc)
                noise = np.random.normal(0, noise_level * np.abs(conc_array).mean(), conc_array.shape)
                noisy_traj['concentrations'][species] = conc_array + noise

            noisy.trajectories.append(noisy_traj)

        noisy_samples.append(noisy)

    return noisy_samples


@torch.no_grad()
def evaluate_tactic_noisy(model, samples: list, device: torch.device, version: str = 'v3') -> float:
    """Evaluate TACTIC on samples."""
    dataset = MultiConditionDataset(samples)

    correct = 0
    total = 0

    for i in range(len(dataset)):
        batch = dataset[i]

        trajectories = batch['trajectories'].unsqueeze(0).to(device)
        conditions = batch['conditions'].unsqueeze(0).to(device)
        condition_mask = batch['condition_mask'].unsqueeze(0).to(device)

        if version == 'v1':
            # v1 uses only 2 features (S, P) - indices 1 and 2 from the 5-feature tensor
            trajectories_v1 = trajectories[:, :, :, 1:3]
            output = model(trajectories_v1, conditions, condition_mask=condition_mask)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Noise Robustness Analysis')
    parser.add_argument('--n-samples', type=int, default=50, help='Samples per mechanism')
    parser.add_argument('--version', type=str, default='v3', choices=['v1', 'v2', 'v3'],
                       help='Model version to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path (default: checkpoints/<version>/best_model.pt)')
    parser.add_argument('--test-set', type=str, default=None, help='Pre-generated test set')
    parser.add_argument('--skip-classical', action='store_true', help='Skip classical baseline')
    parser.add_argument('--save-results', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default checkpoint based on version
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/{args.version}/best_model.pt'

    print("="*70)
    print(f"EXPERIMENT 3: Noise Robustness Analysis ({args.version.upper()})")
    print("="*70)
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Load or generate CLEAN test set (low noise)
    if args.test_set:
        print(f"Loading test set: {args.test_set}")
        samples, _ = load_dataset(args.test_set)
    else:
        print(f"Generating clean test set ({args.n_samples} per mechanism)...")
        config = MultiConditionConfig(n_conditions_per_sample=20, noise_level=0.01)
        generator = MultiConditionGenerator(config, seed=34567)
        samples = generator.generate_batch(args.n_samples, n_workers=4)

    print(f"Test samples: {len(samples)}")

    # Load model
    checkpoint_path = base_dir / args.checkpoint
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Loaded model from: {checkpoint_path}")

    # Noise levels to test
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    results = []

    print("\n" + "="*70)
    print("RESULTS: Accuracy vs Noise Level")
    print("="*70)

    if args.skip_classical or not HAS_CLASSICAL:
        print(f"{'Noise Level':>12} {'TACTIC':>12} {'Degradation':>14}")
        print("-"*40)
    else:
        print(f"{'Noise Level':>12} {'TACTIC':>12} {'Classical':>12} {'TACTIC Deg.':>12}")
        print("-"*55)

    baseline_tactic = None
    baseline_classical = None

    for noise in noise_levels:
        print(f"Evaluating at noise level {noise:.2f}...")

        # Add noise to samples
        noisy_samples = add_noise_to_samples(samples, noise, seed=42)

        tactic_acc = evaluate_tactic_noisy(model, noisy_samples, device, version=args.version)

        if baseline_tactic is None:
            baseline_tactic = tactic_acc

        if args.skip_classical or not HAS_CLASSICAL:
            classical_acc = None
            degradation = (baseline_tactic - tactic_acc) * 100
            print(f"{noise:>12.2f} {tactic_acc*100:>11.1f}% {degradation:>+13.1f}%")
        else:
            classical_results = evaluate_classical_baseline(noisy_samples, criterion='aic')
            classical_acc = classical_results['accuracy']
            if baseline_classical is None:
                baseline_classical = classical_acc
            degradation = (baseline_tactic - tactic_acc) * 100
            print(f"{noise:>12.2f} {tactic_acc*100:>11.1f}% {classical_acc*100:>11.1f}% {degradation:>+11.1f}%")

        results.append({
            'noise_level': noise,
            'tactic_accuracy': tactic_acc,
            'classical_accuracy': classical_acc,
            'tactic_degradation': (baseline_tactic - tactic_acc) if baseline_tactic else 0,
        })

    # Summary
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    acc_clean = results[0]['tactic_accuracy']
    acc_5pct = next((r['tactic_accuracy'] for r in results if r['noise_level'] == 0.05), None)
    acc_20pct = next((r['tactic_accuracy'] for r in results if r['noise_level'] == 0.20), None)

    print(f"Clean data (0% noise):  {acc_clean*100:.1f}%")
    if acc_5pct:
        print(f"Moderate noise (5%):    {acc_5pct*100:.1f}% (degradation: {(acc_clean-acc_5pct)*100:.1f}%)")
    if acc_20pct:
        print(f"High noise (20%):       {acc_20pct*100:.1f}% (degradation: {(acc_clean-acc_20pct)*100:.1f}%)")

    # Find noise level where accuracy drops below 50%
    for r in results:
        if r['tactic_accuracy'] < 0.5:
            print(f"\nAccuracy drops below 50% at noise level: {r['noise_level']:.0%}")
            break
    else:
        print(f"\nAccuracy stays above 50% even at {results[-1]['noise_level']:.0%} noise")

    print("\nTACTIC trained on 3% noise, so moderate robustness expected.")

    # Save results
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"noise_robustness_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump({
                'version': args.version,
                'results': results,
                'n_samples': len(samples),
                'noise_levels_tested': noise_levels,
            }, f, indent=2)
        print(f"\nSaved results to: {output_path}")

    return results


if __name__ == "__main__":
    main()
