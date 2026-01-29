#!/usr/bin/env python
"""
Validate TACTIC on Diverse Mechanism Types

Test TACTIC's ability to classify different enzyme mechanisms using
datasets simulated from literature kinetic parameters.

This provides a more rigorous validation than existing real data
(which is mostly MM-I) by testing across all 10 mechanism types.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tactic_kinetics.models.multi_condition_classifier import (
    create_multi_task_model,
    create_multi_condition_model,
    create_basic_multi_condition_model,
)
from tactic_kinetics.training.multi_condition_dataset import V1Dataset

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

MECHANISM_SHORT = {
    'michaelis_menten_irreversible': 'MM-I',
    'michaelis_menten_reversible': 'MM-R',
    'competitive_inhibition': 'Comp',
    'uncompetitive_inhibition': 'Uncomp',
    'mixed_inhibition': 'Mixed',
    'substrate_inhibition': 'SubInh',
    'ordered_bi_bi': 'OrdBB',
    'random_bi_bi': 'RandBB',
    'ping_pong': 'PP',
    'product_inhibition': 'ProdInh',
}

# Mechanism families for family-level accuracy
MECHANISM_FAMILIES = {
    'simple': ['michaelis_menten_irreversible'],
    'reversible': ['michaelis_menten_reversible', 'product_inhibition'],
    'inhibited': ['competitive_inhibition', 'uncompetitive_inhibition', 'mixed_inhibition'],
    'substrate_regulated': ['substrate_inhibition'],
    'bisubstrate': ['ordered_bi_bi', 'random_bi_bi', 'ping_pong'],
}


def load_dataset(json_path: Path) -> Dict:
    """Load a dataset from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def dataset_to_tactic_input(dataset: Dict, device: torch.device,
                            n_timepoints: int = 20, max_conditions: int = 20) -> Dict:
    """Convert dataset to TACTIC model input tensors."""
    trajectories_list = dataset['trajectories']
    conditions_list = dataset['conditions']

    all_trajectories = []
    all_conditions = []
    all_derived = []

    # Compute normalization statistics
    all_S = []
    all_t = []
    for traj in trajectories_list:
        all_S.extend(traj['S'])
        all_t.extend(traj['t'])

    S_mean = np.mean(all_S) if all_S else 1.0
    S_std = np.std(all_S) if all_S else 1.0
    t_max = max(all_t) if all_t else 1.0

    for traj, cond in zip(trajectories_list, conditions_list):
        t = np.array(traj['t'])
        S = np.array(traj['S'])
        P = np.array(traj.get('P', np.zeros_like(S)))

        S0 = cond['S0']
        I0 = cond.get('I0', 0)
        B0 = cond.get('B0', 0)
        E0 = cond.get('E0', 1e-3)

        # Resample to fixed timepoints
        t_new = np.linspace(t.min(), t.max(), n_timepoints)
        S_new = np.interp(t_new, t, S)
        P_new = np.interp(t_new, t, P)

        # Normalize
        t_norm = t_new / (t_max + 1e-8)
        S_norm = (S_new - S_mean) / (S_std + 1e-8)
        P_norm = (P_new - S_mean) / (S_std + 1e-8)

        # Compute rates
        dS_dt = np.gradient(S_new, t_new)
        dP_dt = np.gradient(P_new, t_new)

        # Trajectory features: [t_norm, S, P, dS/dt, dP/dt]
        traj_feat = np.stack([t_norm, S_norm, P_norm, dS_dt, dP_dt], axis=-1)
        all_trajectories.append(traj_feat)

        # Condition features [log(S0), log(I0), log(B0), log(P0), log(E0), T_norm, pH_norm, type]
        cond_feat = np.zeros(8)
        cond_feat[0] = np.log10(max(S0, 1e-9))
        cond_feat[1] = np.log10(max(I0, 1e-9)) if I0 > 0 else -9.0
        cond_feat[2] = np.log10(max(B0, 1e-9)) if B0 > 0 else -9.0
        cond_feat[3] = -9.0  # No initial product
        cond_feat[4] = np.log10(max(E0, 1e-12))
        T = cond.get('T', 298.15)
        cond_feat[5] = (T - 298.15) / 20.0
        pH = cond.get('pH', 7.0)
        cond_feat[6] = (pH - 7.0) / 1.0
        cond_feat[7] = 0.0  # Substrate variation
        all_conditions.append(cond_feat)

        # Derived features
        derived = np.zeros(8)
        if len(t_new) > 1:
            v0 = abs(dS_dt[0])
            derived[0] = np.log10(max(v0, 1e-10))
        S_frac = S_new / (S_new[0] + 1e-10)
        half_idx = np.searchsorted(-S_frac, -0.5)
        if half_idx < len(t_new):
            derived[1] = np.log10(max(t_new[half_idx], 1.0))
        if len(t_new) > 2:
            v_start = abs(dS_dt[0])
            v_end = abs(dS_dt[-1])
            derived[2] = v_end / (v_start + 1e-10)
        derived[3] = 1 - S_new[-1] / (S_new[0] + 1e-10) if S_new[0] > 1e-10 else 0
        derived[4] = np.log10(max(S0, 1e-9))
        all_derived.append(derived)

    n_total = len(all_trajectories)
    n_conditions = min(n_total, max_conditions)

    trajectories = np.zeros((max_conditions, n_timepoints, 5))
    conditions = np.zeros((max_conditions, 8))
    derived_features = np.zeros((max_conditions, 8))
    condition_mask = np.ones(max_conditions, dtype=bool)

    for i in range(n_conditions):
        trajectories[i] = all_trajectories[i]
        conditions[i] = all_conditions[i]
        derived_features[i] = all_derived[i]
        condition_mask[i] = False

    return {
        'trajectories': torch.tensor(trajectories, dtype=torch.float32).unsqueeze(0).to(device),
        'conditions': torch.tensor(conditions, dtype=torch.float32).unsqueeze(0).to(device),
        'derived_features': torch.tensor(derived_features, dtype=torch.float32).unsqueeze(0).to(device),
        'condition_mask': torch.tensor(condition_mask, dtype=torch.bool).unsqueeze(0).to(device),
        'n_conditions_used': n_conditions,
    }


def get_mechanism_family(mechanism: str) -> str:
    """Get the family for a mechanism."""
    for family, members in MECHANISM_FAMILIES.items():
        if mechanism in members:
            return family
    return 'unknown'


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
    else:
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
def evaluate_dataset(model, dataset: Dict, device: torch.device) -> Dict:
    """Evaluate TACTIC on a single dataset."""
    expected = dataset['mechanism']
    expected_idx = MECHANISMS.index(expected)

    inputs = dataset_to_tactic_input(dataset, device, max_conditions=20)

    output = model(
        inputs['trajectories'],
        inputs['conditions'],
        derived_features=inputs['derived_features'],
        condition_mask=inputs['condition_mask'],
    )

    logits = output['logits']
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_mech = MECHANISMS[pred_idx]

    # Family-level prediction
    expected_family = get_mechanism_family(expected)
    pred_family = get_mechanism_family(pred_mech)

    return {
        'name': dataset['name'],
        'expected': expected,
        'expected_family': expected_family,
        'predicted': pred_mech,
        'predicted_family': pred_family,
        'confidence': float(probs[pred_idx]),
        'correct': pred_mech == expected,
        'family_correct': pred_family == expected_family,
        'all_probs': {MECHANISMS[i]: float(probs[i]) for i in range(10)},
        'n_conditions': inputs['n_conditions_used'],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate TACTIC on diverse mechanisms')
    parser.add_argument('--version', type=str, default='v3', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--save-results', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = Path(args.data_dir) if args.data_dir else base_dir / 'data' / 'real' / 'by_mechanism'

    print("=" * 70)
    print(f"MECHANISM DIVERSITY VALIDATION ({args.version.upper()})")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")

    # Load model
    checkpoint_path = base_dir / 'checkpoints' / args.version / 'best_model.pt'
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Find all datasets
    datasets = []
    for mech_dir in data_dir.iterdir():
        if mech_dir.is_dir():
            for json_file in mech_dir.glob('*.json'):
                datasets.append(json_file)

    print(f"\nFound {len(datasets)} datasets")

    # Evaluate each dataset
    results = []
    for json_path in sorted(datasets):
        dataset = load_dataset(json_path)
        result = evaluate_dataset(model, dataset, device)
        results.append(result)

        status = '✓' if result['correct'] else '✗'
        family_status = '✓' if result['family_correct'] else '✗'
        print(f"\n{status} {result['name']}")
        print(f"    Expected:  {MECHANISM_SHORT.get(result['expected'], result['expected'])}")
        print(f"    Predicted: {MECHANISM_SHORT.get(result['predicted'], result['predicted'])} ({result['confidence']*100:.1f}%)")
        print(f"    Family:    {result['expected_family']} → {result['predicted_family']} {family_status}")

        # Top-3 predictions
        sorted_probs = sorted(result['all_probs'].items(), key=lambda x: -x[1])
        print("    Top-3:")
        for mech, prob in sorted_probs[:3]:
            marker = " ←" if mech == result['expected'] else ""
            print(f"      {MECHANISM_SHORT.get(mech, mech):8s}: {prob*100:5.1f}%{marker}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_total = len(results)
    n_correct = sum(r['correct'] for r in results)
    n_family_correct = sum(r['family_correct'] for r in results)

    print(f"\nMechanism-level accuracy: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
    print(f"Family-level accuracy:    {n_family_correct}/{n_total} ({n_family_correct/n_total*100:.1f}%)")

    # Per-mechanism breakdown
    print("\nPer-mechanism results:")
    print(f"{'Mechanism':<25} {'Expected':<10} {'Predicted':<10} {'Conf':<8} {'Status':<8}")
    print("-" * 70)
    for r in results:
        exp = MECHANISM_SHORT.get(r['expected'], r['expected'][:10])
        pred = MECHANISM_SHORT.get(r['predicted'], r['predicted'][:10])
        status = 'CORRECT' if r['correct'] else 'WRONG'
        print(f"{r['name'][:24]:<25} {exp:<10} {pred:<10} {r['confidence']*100:5.1f}%  {status:<8}")

    # Confusion by family
    print("\nFamily confusion:")
    family_results = {}
    for r in results:
        key = (r['expected_family'], r['predicted_family'])
        family_results[key] = family_results.get(key, 0) + 1
    for (exp, pred), count in sorted(family_results.items()):
        status = '✓' if exp == pred else '✗'
        print(f"  {exp:20s} → {pred:20s}: {count} {status}")

    # Save results
    if args.save_results:
        results_dir = base_dir / 'results' / 'experiments'
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"mechanism_diversity_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            'version': args.version,
            'n_datasets': n_total,
            'mechanism_accuracy': n_correct / n_total,
            'family_accuracy': n_family_correct / n_total,
            'results': results,
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to: {output_path}")


if __name__ == '__main__':
    main()
