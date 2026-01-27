#!/usr/bin/env python
"""
Experiment 4: Family-Level Accuracy

Report accuracy when collapsing 10 mechanisms to 5 families.
Shows practical utility even when exact subtype is wrong.
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

# Mechanism families (biochemically related)
FAMILIES = {
    'simple': ['michaelis_menten_irreversible'],
    'reversible': ['michaelis_menten_reversible', 'product_inhibition'],
    'inhibited': ['competitive_inhibition', 'uncompetitive_inhibition', 'mixed_inhibition'],
    'substrate_regulated': ['substrate_inhibition'],
    'bisubstrate': ['ordered_bi_bi', 'random_bi_bi', 'ping_pong'],
}

# Create reverse mapping
MECHANISM_TO_FAMILY = {}
for family, members in FAMILIES.items():
    for mech in members:
        MECHANISM_TO_FAMILY[mech] = family


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


def to_family(mechanism_name: str) -> str:
    """Convert mechanism name to family."""
    return MECHANISM_TO_FAMILY.get(mechanism_name, 'unknown')


@torch.no_grad()
def evaluate_with_families(model, samples: list, device: torch.device, version: str = 'v3') -> dict:
    """Evaluate TACTIC and compute both mechanism and family accuracy."""
    dataset = MultiConditionDataset(samples)

    all_preds = []
    all_labels = []

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

        all_preds.append(pred_idx)
        all_labels.append(true_idx)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Mechanism-level accuracy
    mechanism_acc = (all_preds == all_labels).mean()

    # Family-level accuracy
    pred_families = [to_family(MECHANISMS[p]) for p in all_preds]
    true_families = [to_family(MECHANISMS[l]) for l in all_labels]
    family_acc = (np.array(pred_families) == np.array(true_families)).mean()

    # Per-family accuracy
    per_family_acc = {}
    per_family_confusion = defaultdict(lambda: defaultdict(int))
    for pred_f, true_f in zip(pred_families, true_families):
        per_family_confusion[true_f][pred_f] += 1

    for family in FAMILIES.keys():
        total = sum(per_family_confusion[family].values())
        correct = per_family_confusion[family][family]
        per_family_acc[family] = correct / total if total > 0 else 0

    # Within-family confusion analysis
    within_family_errors = 0
    between_family_errors = 0
    for pred, true in zip(all_preds, all_labels):
        if pred != true:
            pred_fam = to_family(MECHANISMS[pred])
            true_fam = to_family(MECHANISMS[true])
            if pred_fam == true_fam:
                within_family_errors += 1
            else:
                between_family_errors += 1

    return {
        'mechanism_accuracy': mechanism_acc,
        'family_accuracy': family_acc,
        'per_family_accuracy': per_family_acc,
        'family_confusion': {k: dict(v) for k, v in per_family_confusion.items()},
        'within_family_errors': within_family_errors,
        'between_family_errors': between_family_errors,
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Family-Level Accuracy Analysis')
    parser.add_argument('--n-samples', type=int, default=100, help='Samples per mechanism')
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
    print(f"EXPERIMENT 4: Family-Level Accuracy ({args.version.upper()})")
    print("="*70)
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    # Print family definitions
    print("\nMECHANISM FAMILIES:")
    for family, members in FAMILIES.items():
        print(f"  {family}: {', '.join(members)}")

    # Load or generate test set
    if args.test_set:
        print(f"\nLoading test set: {args.test_set}")
        samples, _ = load_dataset(args.test_set)
    else:
        print(f"\nGenerating test set ({args.n_samples} per mechanism)...")
        config = MultiConditionConfig(n_conditions_per_sample=20)
        generator = MultiConditionGenerator(config, seed=45678)
        samples = generator.generate_batch(args.n_samples, n_workers=4)

    print(f"Test samples: {len(samples)}")

    # Load model
    checkpoint_path = base_dir / args.checkpoint
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Loaded model from: {checkpoint_path}")

    # Evaluate TACTIC
    print("\nEvaluating TACTIC...")
    tactic_results = evaluate_with_families(model, samples, device, version=args.version)

    # Evaluate Classical (if available)
    classical_results = None
    if not args.skip_classical and HAS_CLASSICAL:
        print("Evaluating Classical...")
        classical_raw = evaluate_classical_baseline(samples, criterion='aic')

        # Compute family accuracy for classical
        pred_families = [to_family(MECHANISMS[p]) for p in classical_raw['predictions']]
        true_families = [to_family(MECHANISMS[l]) for l in classical_raw['labels']]
        classical_family_acc = (np.array(pred_families) == np.array(true_families)).mean()

        classical_results = {
            'mechanism_accuracy': classical_raw['accuracy'],
            'family_accuracy': classical_family_acc,
        }

    # Results
    print("\n" + "="*70)
    print("RESULTS: 10-Class vs 5-Family Accuracy")
    print("="*70)

    print(f"\n{'Method':<15} {'10-Class':>12} {'5-Family':>12} {'Improvement':>14}")
    print("-"*55)

    t_mech = tactic_results['mechanism_accuracy']
    t_fam = tactic_results['family_accuracy']
    print(f"{'TACTIC':<15} {t_mech*100:>11.1f}% {t_fam*100:>11.1f}% {(t_fam-t_mech)*100:>+13.1f}%")

    if classical_results:
        c_mech = classical_results['mechanism_accuracy']
        c_fam = classical_results['family_accuracy']
        print(f"{'Classical':<15} {c_mech*100:>11.1f}% {c_fam*100:>11.1f}% {(c_fam-c_mech)*100:>+13.1f}%")

    print(f"{'Random':<15} {'10.0%':>12} {'20.0%':>12} {'+10.0%':>14}")

    # Per-family breakdown
    print("\n" + "-"*70)
    print("PER-FAMILY ACCURACY (TACTIC)")
    print("-"*70)
    for family in FAMILIES.keys():
        acc = tactic_results['per_family_accuracy'][family]
        n_mechs = len(FAMILIES[family])
        print(f"  {family:<20} {acc*100:>6.1f}%  ({n_mechs} mechanisms)")

    # Error analysis
    print("\n" + "-"*70)
    print("ERROR ANALYSIS")
    print("-"*70)
    total_errors = tactic_results['within_family_errors'] + tactic_results['between_family_errors']
    if total_errors > 0:
        print(f"Total errors: {total_errors}")
        print(f"  Within-family errors: {tactic_results['within_family_errors']} ({tactic_results['within_family_errors']/total_errors*100:.1f}%)")
        print(f"  Between-family errors: {tactic_results['between_family_errors']} ({tactic_results['between_family_errors']/total_errors*100:.1f}%)")
        print("\nMost errors are within-family (harder to distinguish subtypes)")

    # Family confusion matrix
    print("\n" + "-"*70)
    print("FAMILY CONFUSION MATRIX")
    print("-"*70)
    families_list = list(FAMILIES.keys())
    true_pred = "True\\Pred"
    header = f"{true_pred:<20}" + "".join(f"{f[:8]:>10}" for f in families_list)
    print(header)
    for true_f in families_list:
        row = f"{true_f:<20}"
        for pred_f in families_list:
            count = tactic_results['family_confusion'].get(true_f, {}).get(pred_f, 0)
            if true_f == pred_f:
                row += f"[{count:>7}]"
            else:
                row += f"{count:>9} "
        print(row)

    # Save results
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"family_accuracy_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output = {
            'version': args.version,
            'tactic': tactic_results,
            'classical': classical_results,
            'families': FAMILIES,
            'n_samples': len(samples),
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        print(f"\nSaved results to: {output_path}")

    return tactic_results, classical_results


if __name__ == "__main__":
    main()
