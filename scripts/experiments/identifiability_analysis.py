#!/usr/bin/env python
"""
Experiment 5: Identifiability Analysis

Empirically verify which mechanism pairs are theoretically confusable
and which can be distinguished with multi-condition experiments.

This formalizes the core insight of the paper:
- Single curves are fundamentally ambiguous
- Multiple conditions enable discrimination
- Some pairs remain confusable even with many conditions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

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

# Theoretical confusability groups (from biochemistry)
CONFUSABILITY_GROUPS = {
    'inhibition_types': {
        'mechanisms': ['competitive_inhibition', 'uncompetitive_inhibition', 'mixed_inhibition'],
        'reason': 'All show inhibitor effects, differ in Lineweaver-Burk patterns',
        'distinguishable_with': 'Varying [I] at multiple [S] levels',
    },
    'bisubstrate_patterns': {
        'mechanisms': ['ordered_bi_bi', 'random_bi_bi', 'ping_pong'],
        'reason': 'Similar two-substrate kinetics, differ in initial velocity patterns',
        'distinguishable_with': 'Varying [A] and [B] independently, product inhibition studies',
    },
    'reversibility': {
        'mechanisms': ['michaelis_menten_reversible', 'product_inhibition'],
        'reason': 'Both show product effects, differ in mechanism',
        'distinguishable_with': 'Equilibrium approach experiments, spiking initial product',
    },
}


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
def compute_confusion_matrix(model, samples: list, device: torch.device, version: str = 'v3') -> np.ndarray:
    """Compute full confusion matrix."""
    dataset = MultiConditionDataset(samples)

    confusion = np.zeros((10, 10), dtype=int)

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
        confusion[true_idx, pred_idx] += 1

    return confusion


def analyze_confusability(confusion: np.ndarray) -> dict:
    """Analyze pairwise confusability from confusion matrix."""
    n = confusion.shape[0]
    pairwise_confusion = {}

    for i in range(n):
        for j in range(n):
            if i != j:
                # Asymmetric confusion rate: how often is i predicted as j?
                total_i = confusion[i].sum()
                if total_i > 0:
                    rate_i_to_j = confusion[i, j] / total_i
                else:
                    rate_i_to_j = 0

                total_j = confusion[j].sum()
                if total_j > 0:
                    rate_j_to_i = confusion[j, i] / total_j
                else:
                    rate_j_to_i = 0

                # Symmetric confusion rate
                symmetric_rate = (rate_i_to_j + rate_j_to_i) / 2

                if symmetric_rate > 0.05:  # Only report significant confusions
                    pair = tuple(sorted([MECHANISMS[i], MECHANISMS[j]]))
                    if pair not in pairwise_confusion or symmetric_rate > pairwise_confusion[pair]['symmetric_rate']:
                        pairwise_confusion[pair] = {
                            'symmetric_rate': symmetric_rate,
                            f'{MECHANISMS[i]}_to_{MECHANISMS[j]}': rate_i_to_j,
                            f'{MECHANISMS[j]}_to_{MECHANISMS[i]}': rate_j_to_i,
                        }

    return pairwise_confusion


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Identifiability Analysis')
    parser.add_argument('--n-samples', type=int, default=100, help='Samples per mechanism')
    parser.add_argument('--version', type=str, default='v3', choices=['v1', 'v2', 'v3'],
                       help='Model version to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path (default: checkpoints/<version>/best_model.pt)')
    parser.add_argument('--test-set', type=str, default=None, help='Pre-generated test set')
    parser.add_argument('--save-results', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default checkpoint based on version
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/{args.version}/best_model.pt'

    print("="*70)
    print(f"EXPERIMENT 5: Identifiability Analysis ({args.version.upper()})")
    print("="*70)
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Print theoretical confusability groups
    print("\n" + "="*70)
    print("THEORETICAL CONFUSABILITY GROUPS")
    print("="*70)
    for group_name, group_info in CONFUSABILITY_GROUPS.items():
        print(f"\n{group_name.upper()}:")
        print(f"  Mechanisms: {', '.join(group_info['mechanisms'])}")
        print(f"  Reason: {group_info['reason']}")
        print(f"  Distinguishable with: {group_info['distinguishable_with']}")

    # Load or generate test set
    if args.test_set:
        print(f"\nLoading test set: {args.test_set}")
        samples, _ = load_dataset(args.test_set)
    else:
        print(f"\nGenerating test set ({args.n_samples} per mechanism)...")
        config = MultiConditionConfig(n_conditions_per_sample=20)
        generator = MultiConditionGenerator(config, seed=56789)
        samples = generator.generate_batch(args.n_samples, n_workers=4)

    print(f"Test samples: {len(samples)}")

    # Load model
    checkpoint_path = base_dir / args.checkpoint
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Loaded model from: {checkpoint_path}")

    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    confusion = compute_confusion_matrix(model, samples, device, version=args.version)

    # Per-mechanism accuracy
    print("\n" + "="*70)
    print("PER-MECHANISM ACCURACY")
    print("="*70)
    print(f"{'Mechanism':<35} {'Accuracy':>10} {'N':>6}")
    print("-"*55)
    for i, mech in enumerate(MECHANISMS):
        total = confusion[i].sum()
        correct = confusion[i, i]
        acc = correct / total if total > 0 else 0
        print(f"{mech:<35} {acc*100:>9.1f}% {total:>6}")

    # Most confused pairs
    print("\n" + "="*70)
    print("MOST CONFUSED MECHANISM PAIRS")
    print("="*70)

    pairwise = analyze_confusability(confusion)
    sorted_pairs = sorted(pairwise.items(), key=lambda x: -x[1]['symmetric_rate'])

    print(f"{'Mechanism 1':<25} {'Mechanism 2':<25} {'Confusion Rate':>15}")
    print("-"*70)
    for pair, rates in sorted_pairs[:15]:
        print(f"{pair[0]:<25} {pair[1]:<25} {rates['symmetric_rate']*100:>14.1f}%")

    # Verify theoretical predictions
    print("\n" + "="*70)
    print("THEORETICAL PREDICTIONS vs EMPIRICAL RESULTS")
    print("="*70)

    for group_name, group_info in CONFUSABILITY_GROUPS.items():
        print(f"\n{group_name.upper()}:")
        mechs = group_info['mechanisms']
        total_confusion = 0
        n_pairs = 0

        for i, m1 in enumerate(mechs):
            for m2 in mechs[i+1:]:
                pair = tuple(sorted([m1, m2]))
                if pair in pairwise:
                    rate = pairwise[pair]['symmetric_rate']
                    total_confusion += rate
                    n_pairs += 1
                    print(f"  {m1} <-> {m2}: {rate*100:.1f}%")
                else:
                    print(f"  {m1} <-> {m2}: <5% (well distinguished)")

        if n_pairs > 0:
            avg = total_confusion / n_pairs
            print(f"  Average within-group confusion: {avg*100:.1f}%")

    # Confusion matrix display
    print("\n" + "="*70)
    print("FULL CONFUSION MATRIX (counts)")
    print("="*70)

    # Short names
    short = [m[:4] for m in MECHANISMS]
    header = "True\\Pred " + " ".join(f"{s:>5}" for s in short)
    print(header)
    for i, mech in enumerate(MECHANISMS):
        row = f"{mech[:9]:<9}"
        for j in range(10):
            if i == j:
                row += f"[{confusion[i,j]:>4}]"
            elif confusion[i, j] > 0:
                row += f" {confusion[i,j]:>4} "
            else:
                row += "    . "
        print(row)

    # Theorem statement
    print("\n" + "="*70)
    print("IDENTIFIABILITY THEOREM (Informal)")
    print("="*70)
    print("""
Theorem 1 (Mechanism Identifiability from Kinetic Data)

Let M = {m_1, ..., m_10} be the set of enzyme mechanisms.
Let D(m, theta, C) denote kinetic trajectories from mechanism m
with parameters theta under conditions C.

(a) Single-condition identifiability:
    For any single condition c, mechanisms in the same family F
    produce trajectories D(m_i, theta_i, c) ≈ D(m_j, theta_j, c)
    for some theta_i, theta_j.

    Empirical support: v0 (single-curve) achieves only 10.6% accuracy.

(b) Multi-condition identifiability:
    Given conditions C = {c_1, ..., c_n} that vary [I] at multiple [S]:
    - Competitive, uncompetitive, mixed become distinguishable

    Given conditions C that vary [A] and [B] independently:
    - Ordered, random, ping_pong become distinguishable (in principle)

    Empirical support: v3 (20 conditions) achieves 62% accuracy.

(c) Residual confusions:
    Some mechanism pairs remain confusable even with many conditions:
    - MM_reversible <-> product_inhibition (both show product effects)
    - ordered_bi_bi <-> random_bi_bi (similar initial velocity patterns)

    These require specialized experiments beyond standard kinetic assays.
""")

    # Save results
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"identifiability_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output = {
            'version': args.version,
            'confusion_matrix': confusion.tolist(),
            'pairwise_confusion': {str(k): v for k, v in pairwise.items()},
            'theoretical_groups': CONFUSABILITY_GROUPS,
            'mechanisms': MECHANISMS,
            'n_samples': len(samples),
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to: {output_path}")

    return confusion, pairwise


if __name__ == "__main__":
    main()
