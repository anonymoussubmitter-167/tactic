#!/usr/bin/env python
"""
Experiment 1: Confidence-Accuracy Analysis

When TACTIC is confident, is it correct?
Shows model calibration - important for practical reliability.
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
def analyze_confidence(model, samples: list, device: torch.device, version: str = 'v3') -> dict:
    """Analyze confidence vs accuracy relationship."""
    dataset = MultiConditionDataset(samples)

    all_confidences = []
    all_correct = []
    all_preds = []
    all_labels = []
    all_probs = []

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

        logits = output['logits']
        probs = F.softmax(logits, dim=-1)
        confidence, pred = probs.max(dim=-1)
        true_idx = batch['mechanism_idx'].item()
        correct = (pred.item() == true_idx)

        all_confidences.append(confidence.item())
        all_correct.append(correct)
        all_preds.append(pred.item())
        all_labels.append(true_idx)
        all_probs.append(probs.cpu().numpy().flatten())

    confidences = np.array(all_confidences)
    correct = np.array(all_correct)

    # Bin by confidence
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    bin_results = []

    print("\n" + "="*60)
    print("CONFIDENCE-ACCURACY CALIBRATION")
    print("="*60)
    print(f"{'Confidence Range':<20} {'Accuracy':>12} {'N Samples':>12}")
    print("-"*60)

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        n = mask.sum()
        if n > 0:
            acc = correct[mask].mean()
            bin_results.append({
                'confidence_low': low,
                'confidence_high': high,
                'accuracy': float(acc),
                'n_samples': int(n),
            })
            print(f"{low:.1f} - {high:.1f}          {acc*100:>10.1f}%  {n:>10d}")
        else:
            print(f"{low:.1f} - {high:.1f}          {'N/A':>10s}  {0:>10d}")

    # Per-mechanism confidence
    print("\n" + "-"*60)
    print("PER-MECHANISM MEAN CONFIDENCE")
    print("-"*60)

    per_mech_conf = {}
    per_mech_acc = {}
    for mech_idx, mech_name in enumerate(MECHANISMS):
        mask = np.array(all_labels) == mech_idx
        if mask.sum() > 0:
            per_mech_conf[mech_name] = float(confidences[mask].mean())
            per_mech_acc[mech_name] = float(correct[mask].mean())
            print(f"{mech_name:<35} conf={per_mech_conf[mech_name]:.3f}  acc={per_mech_acc[mech_name]*100:.1f}%")

    # Overall stats
    print("\n" + "-"*60)
    print("SUMMARY STATISTICS")
    print("-"*60)
    print(f"Mean confidence: {confidences.mean():.3f}")
    print(f"Median confidence: {np.median(confidences):.3f}")
    print(f"Overall accuracy: {correct.mean()*100:.1f}%")
    print(f"High confidence (>0.8) accuracy: {correct[confidences > 0.8].mean()*100:.1f}%")
    print(f"Low confidence (<0.4) accuracy: {correct[confidences < 0.4].mean()*100:.1f}%")

    # Expected Calibration Error (ECE)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correct[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= len(confidences)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    return {
        'bin_results': bin_results,
        'per_mechanism_confidence': per_mech_conf,
        'per_mechanism_accuracy': per_mech_acc,
        'mean_confidence': float(confidences.mean()),
        'median_confidence': float(np.median(confidences)),
        'overall_accuracy': float(correct.mean()),
        'ece': float(ece),
        'n_samples': len(samples),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Confidence-Accuracy Analysis')
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

    print("="*60)
    print(f"EXPERIMENT 1: Confidence-Accuracy Analysis ({args.version.upper()})")
    print("="*60)
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Load or generate test set
    if args.test_set:
        print(f"Loading test set: {args.test_set}")
        samples, _ = load_dataset(args.test_set)
    else:
        print(f"Generating test set ({args.n_samples} per mechanism)...")
        config = MultiConditionConfig(n_conditions_per_sample=20)
        generator = MultiConditionGenerator(config, seed=12345)
        samples = generator.generate_batch(args.n_samples, n_workers=4)

    print(f"Test samples: {len(samples)}")

    # Load model
    checkpoint_path = base_dir / args.checkpoint
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Loaded model from: {checkpoint_path}")

    # Run analysis
    results = analyze_confidence(model, samples, device, version=args.version)
    results['version'] = args.version

    # Save results
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"confidence_analysis_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {output_path}")

    return results


if __name__ == "__main__":
    main()
