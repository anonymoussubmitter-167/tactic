#!/usr/bin/env python
"""
Experiment 6: Error Correlation Analysis

When TACTIC fails, does Classical also fail? (And vice versa)
Shows complementary strengths between methods.
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

# Import classical baseline
sys.path.insert(0, str(Path(__file__).parent.parent))
from classical_baseline import evaluate_classical_baseline

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
def get_tactic_predictions(model, samples: list, device: torch.device, version: str = 'v3') -> tuple:
    """Get TACTIC predictions."""
    dataset = MultiConditionDataset(samples)

    predictions = []
    labels = []
    confidences = []

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

        probs = F.softmax(output['logits'], dim=-1)
        conf, pred = probs.max(dim=-1)

        predictions.append(pred.item())
        labels.append(batch['mechanism_idx'].item())
        confidences.append(conf.item())

    return np.array(predictions), np.array(labels), np.array(confidences)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Error Correlation Analysis')
    parser.add_argument('--n-samples', type=int, default=50, help='Samples per mechanism')
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
    print(f"EXPERIMENT 6: Error Correlation Analysis ({args.version.upper()})")
    print("="*70)
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Load or generate test set
    if args.test_set:
        print(f"Loading test set: {args.test_set}")
        samples, _ = load_dataset(args.test_set)
    else:
        print(f"Generating test set ({args.n_samples} per mechanism)...")
        config = MultiConditionConfig(n_conditions_per_sample=20)
        generator = MultiConditionGenerator(config, seed=67890)
        samples = generator.generate_batch(args.n_samples, n_workers=4)

    print(f"Test samples: {len(samples)}")

    # Load TACTIC model
    checkpoint_path = base_dir / args.checkpoint
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Loaded model from: {checkpoint_path}")

    # Get TACTIC predictions
    print("\nRunning TACTIC inference...")
    tactic_preds, labels, tactic_conf = get_tactic_predictions(model, samples, device, version=args.version)
    tactic_correct = (tactic_preds == labels)

    # Get Classical predictions
    print("Running Classical inference (this may take a while)...")
    classical_results = evaluate_classical_baseline(samples, criterion='aic')
    classical_preds = np.array(classical_results['predictions'])
    classical_correct = (classical_preds == labels)

    # Error correlation analysis
    print("\n" + "="*70)
    print("ERROR CORRELATION ANALYSIS")
    print("="*70)

    both_correct = (tactic_correct & classical_correct).sum()
    tactic_only = (tactic_correct & ~classical_correct).sum()
    classical_only = (~tactic_correct & classical_correct).sum()
    both_wrong = (~tactic_correct & ~classical_correct).sum()
    total = len(labels)

    print(f"""
    ┌─────────────────┬─────────────────┬─────────────────┐
    │                 │ Classical ✓     │ Classical ✗     │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ TACTIC ✓        │ {both_correct:4d} ({both_correct/total*100:4.1f}%)    │ {tactic_only:4d} ({tactic_only/total*100:4.1f}%)    │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ TACTIC ✗        │ {classical_only:4d} ({classical_only/total*100:4.1f}%)    │ {both_wrong:4d} ({both_wrong/total*100:4.1f}%)    │
    └─────────────────┴─────────────────┴─────────────────┘
    """)

    print(f"TACTIC accuracy:    {tactic_correct.mean()*100:.1f}%")
    print(f"Classical accuracy: {classical_correct.mean()*100:.1f}%")

    # Disagreement analysis
    disagree_mask = tactic_preds != classical_preds
    n_disagree = disagree_mask.sum()

    print(f"\n" + "-"*70)
    print(f"DISAGREEMENT ANALYSIS")
    print("-"*70)
    print(f"Methods disagree on: {n_disagree} samples ({n_disagree/total*100:.1f}%)")

    if n_disagree > 0:
        tactic_wins_on_disagree = tactic_correct[disagree_mask].mean()
        classical_wins_on_disagree = classical_correct[disagree_mask].mean()
        print(f"When they disagree:")
        print(f"  TACTIC is correct:    {tactic_wins_on_disagree*100:.1f}%")
        print(f"  Classical is correct: {classical_wins_on_disagree*100:.1f}%")

    # Per-mechanism analysis
    print(f"\n" + "-"*70)
    print(f"PER-MECHANISM: WHO WINS?")
    print("-"*70)
    print(f"{'Mechanism':<35} {'TACTIC':>8} {'Classic':>8} {'Winner':>10}")
    print("-"*70)

    tactic_wins_mech = 0
    classical_wins_mech = 0
    ties = 0

    for mech_idx, mech_name in enumerate(MECHANISMS):
        mask = labels == mech_idx
        if mask.sum() == 0:
            continue

        t_acc = tactic_correct[mask].mean()
        c_acc = classical_correct[mask].mean()

        if t_acc > c_acc + 0.05:
            winner = "TACTIC"
            tactic_wins_mech += 1
        elif c_acc > t_acc + 0.05:
            winner = "Classical"
            classical_wins_mech += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"{mech_name:<35} {t_acc*100:>7.1f}% {c_acc*100:>7.1f}% {winner:>10}")

    print("-"*70)
    print(f"TACTIC wins: {tactic_wins_mech}, Classical wins: {classical_wins_mech}, Ties: {ties}")

    # Ensemble potential
    print(f"\n" + "-"*70)
    print(f"ENSEMBLE POTENTIAL")
    print("-"*70)

    # Simple voting ensemble
    ensemble_correct = ((tactic_preds == labels) | (classical_preds == labels))
    max_ensemble_acc = ensemble_correct.mean()
    print(f"Upper bound (either correct): {max_ensemble_acc*100:.1f}%")

    # Confidence-weighted ensemble
    # When TACTIC is confident (>0.8), use TACTIC; else use classical
    high_conf_mask = tactic_conf > 0.8
    conf_ensemble_preds = np.where(high_conf_mask, tactic_preds, classical_preds)
    conf_ensemble_acc = (conf_ensemble_preds == labels).mean()
    print(f"Confidence-weighted (TACTIC if conf>0.8): {conf_ensemble_acc*100:.1f}%")

    # When TACTIC is low confidence (<0.5), use classical
    low_conf_mask = tactic_conf < 0.5
    conservative_preds = np.where(low_conf_mask, classical_preds, tactic_preds)
    conservative_acc = (conservative_preds == labels).mean()
    print(f"Conservative (Classical if TACTIC conf<0.5): {conservative_acc*100:.1f}%")

    # Correlation coefficient
    correlation = np.corrcoef(tactic_correct.astype(float), classical_correct.astype(float))[0, 1]
    print(f"\nError correlation coefficient: {correlation:.3f}")
    if correlation < 0.5:
        print("Low correlation suggests methods have complementary strengths!")
    else:
        print("High correlation suggests methods fail on similar samples.")

    # Save results
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"error_correlation_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output = {
            'contingency_table': {
                'both_correct': int(both_correct),
                'tactic_only': int(tactic_only),
                'classical_only': int(classical_only),
                'both_wrong': int(both_wrong),
            },
            'tactic_accuracy': float(tactic_correct.mean()),
            'classical_accuracy': float(classical_correct.mean()),
            'disagreement_rate': float(n_disagree / total),
            'correlation': float(correlation),
            'ensemble_upper_bound': float(max_ensemble_acc),
            'n_samples': len(samples),
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to: {output_path}")

    return {
        'both_correct': both_correct,
        'tactic_only': tactic_only,
        'classical_only': classical_only,
        'both_wrong': both_wrong,
        'correlation': correlation,
    }


if __name__ == "__main__":
    main()
