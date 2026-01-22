#!/usr/bin/env python
"""
Compare TACTIC vs Classical (AIC/BIC) Model Selection

This script runs both methods on the SAME test set to enable fair comparison.
This is the key experiment for the paper - we need to beat this baseline.

Usage:
    # Quick test (10 samples/mechanism, ~5 min)
    python scripts/compare_methods.py --n-samples 10

    # Full comparison (100 samples/mechanism, ~1 hour)
    python scripts/compare_methods.py --n-samples 100

    # Paper-quality (200 samples/mechanism, ~2 hours)
    python scripts/compare_methods.py --n-samples 200 --save-results
"""

import numpy as np
import torch
from pathlib import Path
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    MultiConditionSample,
    save_dataset,
    load_dataset,
)
from tactic_kinetics.training.multi_condition_dataset import MultiConditionDataset
from tactic_kinetics.models.multi_condition_classifier import (
    create_multi_task_model,
)

from classical_baseline import (
    ClassicalModelSelector,
    evaluate_classical_baseline,
    MECHANISMS,
)


def load_tactic_model(checkpoint_path: Path, device: torch.device):
    """Load trained TACTIC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = create_multi_task_model(
        d_model=128,
        n_heads=4,
        n_traj_layers=2,
        n_cross_layers=3,
        n_mechanisms=10,
        dropout=0.0,  # No dropout for inference
    )

    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate_tactic(model, samples: list, device: torch.device, verbose: bool = True) -> dict:
    """Evaluate TACTIC model on samples."""

    # Create dataset
    dataset = MultiConditionDataset(samples)

    correct = 0
    total = 0
    per_mechanism = {m: {'correct': 0, 'total': 0} for m in MECHANISMS}
    confusion = {m1: {m2: 0 for m2 in MECHANISMS} for m1 in MECHANISMS}

    for i in range(len(dataset)):
        if verbose and i % 100 == 0:
            print(f"  TACTIC inference: {i}/{len(dataset)}")

        batch = dataset[i]

        # Prepare inputs
        trajectories = batch['trajectories'].unsqueeze(0).to(device)
        conditions = batch['conditions'].unsqueeze(0).to(device)
        derived_features = batch['derived_features'].unsqueeze(0).to(device)
        condition_mask = batch['condition_mask'].unsqueeze(0).to(device)

        # Run inference
        output = model(
            trajectories,
            conditions,
            derived_features=derived_features,
            condition_mask=condition_mask,
        )

        logits = output['logits']
        pred_idx = logits.argmax(dim=-1).item()
        true_idx = batch['mechanism_idx'].item()

        true_mech = MECHANISMS[true_idx]
        pred_mech = MECHANISMS[pred_idx]

        per_mechanism[true_mech]['total'] += 1
        confusion[true_mech][pred_mech] += 1

        if pred_idx == true_idx:
            correct += 1
            per_mechanism[true_mech]['correct'] += 1

        total += 1

    accuracy = correct / total if total > 0 else 0

    per_mech_acc = {}
    for m in MECHANISMS:
        if per_mechanism[m]['total'] > 0:
            per_mech_acc[m] = per_mechanism[m]['correct'] / per_mechanism[m]['total']
        else:
            per_mech_acc[m] = 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_mechanism': per_mech_acc,
        'confusion': confusion,
    }


def print_comparison_results(tactic_results: dict, classical_results: dict):
    """Print formatted comparison results."""

    print("\n" + "="*80)
    print("COMPARISON RESULTS: TACTIC vs Classical (AIC/BIC)")
    print("="*80)

    # Overall accuracy
    print("\n" + "-"*80)
    print("OVERALL ACCURACY")
    print("-"*80)
    print(f"  TACTIC:          {tactic_results['accuracy']*100:5.1f}%  ({tactic_results['correct']}/{tactic_results['total']})")
    print(f"  Classical (AIC): {classical_results['accuracy']*100:5.1f}%  ({classical_results['correct']}/{classical_results['total']})")

    diff = tactic_results['accuracy'] - classical_results['accuracy']
    winner = "TACTIC" if diff > 0 else "Classical" if diff < 0 else "Tie"
    print(f"\n  Difference: {diff*100:+.1f}% ({winner} wins)")

    # Per-mechanism comparison
    print("\n" + "-"*80)
    print("PER-MECHANISM ACCURACY")
    print("-"*80)
    print(f"{'Mechanism':35s} {'TACTIC':>8s} {'Classical':>10s} {'Diff':>8s} {'Winner':>10s}")
    print("-"*80)

    tactic_wins = 0
    classical_wins = 0
    ties = 0

    for mech in MECHANISMS:
        t_acc = tactic_results['per_mechanism'].get(mech, 0)
        c_acc = classical_results['per_mechanism'].get(mech, 0)
        diff = t_acc - c_acc

        if diff > 0.01:
            winner = "TACTIC"
            tactic_wins += 1
        elif diff < -0.01:
            winner = "Classical"
            classical_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"{mech:35s} {t_acc*100:7.1f}% {c_acc*100:9.1f}% {diff*100:+7.1f}% {winner:>10s}")

    print("-"*80)
    print(f"TACTIC wins: {tactic_wins}/10, Classical wins: {classical_wins}/10, Ties: {ties}/10")

    # Summary for paper
    print("\n" + "="*80)
    print("SUMMARY FOR PAPER")
    print("="*80)
    print(f"""
    Method          | Accuracy | Notes
    ----------------|----------|-------------------------------
    Random baseline |   10.0%  | 1/10 classes
    Classical (AIC) | {classical_results['accuracy']*100:5.1f}%  | Traditional model selection
    TACTIC (ML)     | {tactic_results['accuracy']*100:5.1f}%  | Our method

    Key finding: {"TACTIC outperforms" if diff > 0 else "Classical outperforms" if diff < 0 else "Methods perform similarly"}
    Difference: {diff*100:+.1f}%
    """)


def print_confusion_matrices(tactic_results: dict, classical_results: dict):
    """Print confusion matrices for both methods."""

    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)

    for method, results in [("TACTIC", tactic_results), ("Classical (AIC)", classical_results)]:
        print(f"\n{method}:")
        print("-"*80)

        # Short names for display
        short_names = [m[:4] for m in MECHANISMS]
        header = "True\\Pred " + " ".join(f"{n:>5s}" for n in short_names)
        print(header)

        for true_mech in MECHANISMS:
            row = f"{true_mech[:9]:9s}"
            for pred_mech in MECHANISMS:
                count = results['confusion'][true_mech][pred_mech]
                if count > 0:
                    row += f" {count:5d}"
                else:
                    row += "     ."
            print(row)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare TACTIC vs Classical baseline')
    parser.add_argument('--n-samples', type=int, default=20,
                       help='Samples per mechanism for test set')
    parser.add_argument('--seed', type=int, default=99999,
                       help='Random seed for test set generation')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to TACTIC checkpoint (default: checkpoints/best_model.pt)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save detailed results to JSON')
    parser.add_argument('--load-test-set', type=str, default=None,
                       help='Load existing test set instead of generating')
    parser.add_argument('--save-test-set', type=str, default=None,
                       help='Save generated test set for reproducibility')
    parser.add_argument('--skip-tactic', action='store_true',
                       help='Skip TACTIC evaluation (only run classical)')
    parser.add_argument('--skip-classical', action='store_true',
                       help='Skip classical evaluation (only run TACTIC)')
    args = parser.parse_args()

    print("="*80)
    print("TACTIC vs Classical Model Selection Comparison")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Samples per mechanism: {args.n_samples}")
    print(f"Total test samples: {args.n_samples * 10}")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    checkpoint_dir = base_dir / "checkpoints"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Load or generate test set
    if args.load_test_set:
        print(f"\nLoading test set from: {args.load_test_set}")
        samples, _ = load_dataset(args.load_test_set)
    else:
        print(f"\nGenerating test set with seed={args.seed}...")
        config = MultiConditionConfig(n_conditions_per_sample=20)
        generator = MultiConditionGenerator(config, seed=args.seed)
        samples = generator.generate_batch(args.n_samples, n_workers=1)
        print(f"Generated {len(samples)} test samples")

        if args.save_test_set:
            save_dataset(samples, args.save_test_set, config)
            print(f"Saved test set to: {args.save_test_set}")

    # ========== TACTIC Evaluation ==========
    tactic_results = None
    tactic_time = 0

    if not args.skip_tactic:
        print("\n" + "-"*80)
        print("Evaluating TACTIC...")
        print("-"*80)

        checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "best_model.pt"
        if not checkpoint_path.exists():
            print(f"WARNING: TACTIC checkpoint not found: {checkpoint_path}")
            print("Skipping TACTIC evaluation. Train a model first with: python train.py --multi-task")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            model = load_tactic_model(checkpoint_path, device)
            print(f"Loaded TACTIC model from: {checkpoint_path}")

            start_time = time.time()
            tactic_results = evaluate_tactic(model, samples, device)
            tactic_time = time.time() - start_time

            print(f"\nTACTIC accuracy: {tactic_results['accuracy']*100:.1f}%")
            print(f"TACTIC time: {tactic_time:.1f}s ({tactic_time/len(samples)*1000:.1f}ms per sample)")

    # ========== Classical Evaluation ==========
    classical_results = None
    classical_time = 0

    if not args.skip_classical:
        print("\n" + "-"*80)
        print("Evaluating Classical (AIC) Model Selection...")
        print("-"*80)

        start_time = time.time()
        classical_results = evaluate_classical_baseline(samples, criterion='aic')
        classical_time = time.time() - start_time

        print(f"\nClassical accuracy: {classical_results['accuracy']*100:.1f}%")
        print(f"Classical time: {classical_time:.1f}s ({classical_time/len(samples):.1f}s per sample)")

    # ========== Comparison ==========
    if tactic_results and classical_results:
        print_comparison_results(tactic_results, classical_results)
        print_confusion_matrices(tactic_results, classical_results)

        # Speed comparison
        print("\n" + "-"*80)
        print("SPEED COMPARISON")
        print("-"*80)
        print(f"  TACTIC:    {tactic_time:7.1f}s total, {tactic_time/len(samples)*1000:6.1f}ms per sample")
        print(f"  Classical: {classical_time:7.1f}s total, {classical_time/len(samples):6.1f}s per sample")
        speedup = classical_time / (tactic_time + 1e-6)
        print(f"\n  TACTIC is {speedup:.0f}x faster than classical fitting")

        # Save results
        if args.save_results:
            results = {
                'date': datetime.now().isoformat(),
                'n_samples': args.n_samples,
                'seed': args.seed,
                'tactic': {
                    'accuracy': tactic_results['accuracy'],
                    'per_mechanism': tactic_results['per_mechanism'],
                    'time_seconds': tactic_time,
                },
                'classical': {
                    'accuracy': classical_results['accuracy'],
                    'per_mechanism': classical_results['per_mechanism'],
                    'time_seconds': classical_time,
                    'criterion': 'aic',
                },
                'speedup': speedup,
            }

            results_path = results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved detailed results to: {results_path}")

    elif tactic_results:
        print(f"\nTACTIC accuracy: {tactic_results['accuracy']*100:.1f}%")
    elif classical_results:
        print(f"\nClassical accuracy: {classical_results['accuracy']*100:.1f}%")

    return tactic_results, classical_results


if __name__ == "__main__":
    main()
