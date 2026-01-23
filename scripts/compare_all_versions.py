#!/usr/bin/env python
"""
Compare all TACTIC versions (v0, v1, v2, v3) and classical baseline.

Generates a comprehensive comparison table for the paper.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    load_dataset,
)
from tactic_kinetics.training.multi_condition_dataset import MultiConditionDataset
from tactic_kinetics.models.multi_condition_classifier import (
    create_single_curve_model,
    create_basic_multi_condition_model,
    create_multi_condition_model,
    create_multi_task_model,
)
from classical_baseline import evaluate_classical_baseline, MECHANISMS


def load_model(checkpoint_path: Path, device: torch.device, version: str = 'v2'):
    """Load a trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if version == 'v0':
        model = create_single_curve_model(
            d_model=128, n_heads=4, n_layers=2,
            n_mechanisms=10, dropout=0.0,
        )
    elif version == 'v1':
        model = create_basic_multi_condition_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )
    elif version == 'v3':
        model = create_multi_task_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )
    else:  # v2
        model = create_multi_condition_model(
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
def evaluate_v0(model, samples: list, device: torch.device) -> dict:
    """Evaluate v0 single-curve model on samples (using first trajectory only)."""
    correct = 0
    total = 0
    per_mechanism = {m: {'correct': 0, 'total': 0} for m in MECHANISMS}

    for sample in samples:
        # Extract first trajectory for single-curve evaluation
        traj = sample.trajectories[0]
        t = traj['t']
        concentrations = traj['concentrations']
        conditions = traj['conditions']

        # Get substrate
        if 'S' in concentrations:
            S = concentrations['S']
        elif 'A' in concentrations:
            S = concentrations['A']
        else:
            S = list(concentrations.values())[0]

        # Interpolate to 20 timepoints
        n_timepoints = 20
        if len(t) != n_timepoints:
            t_new = np.linspace(t[0], t[-1], n_timepoints)
            S = np.interp(t_new, t, S)
            t = t_new

        # Normalize
        t_norm = t / (t.max() + 1e-8)
        S_norm = (S - S.mean()) / (S.std() + 1e-8)

        # Prepare inputs
        times = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(0).to(device)
        values = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0).to(device)

        # Condition features (4 features: T, pH, S0, E0)
        cond_features = np.zeros(4)
        cond_features[0] = (conditions.get('T', 298.15) - 298.15) / 20.0
        cond_features[1] = (conditions.get('pH', 7.0) - 7.0) / 2.0
        S0 = conditions.get('S0', conditions.get('A0', 1.0))
        cond_features[2] = np.log10(max(S0, 1e-9))
        cond_features[3] = np.log10(max(conditions.get('E0', 1e-3), 1e-12))

        cond_tensor = torch.tensor(cond_features, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.zeros(1, n_timepoints, dtype=torch.bool).to(device)

        output = model(times, values, cond_tensor, mask)
        pred_idx = output['logits'].argmax(dim=-1).item()
        true_idx = sample.mechanism_idx

        true_mech = MECHANISMS[true_idx]
        per_mechanism[true_mech]['total'] += 1
        if pred_idx == true_idx:
            correct += 1
            per_mechanism[true_mech]['correct'] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    per_mech_acc = {
        m: per_mechanism[m]['correct'] / per_mechanism[m]['total']
        if per_mechanism[m]['total'] > 0 else 0
        for m in MECHANISMS
    }

    return {'accuracy': accuracy, 'per_mechanism': per_mech_acc}


@torch.no_grad()
def evaluate_v1(model, samples: list, device: torch.device) -> dict:
    """Evaluate v1 basic multi-condition model (2 features, no derived)."""
    correct = 0
    total = 0
    per_mechanism = {m: {'correct': 0, 'total': 0} for m in MECHANISMS}
    max_conditions = 5
    n_timepoints = 20

    # Compute normalization stats
    all_conc, all_times = [], []
    for sample in samples:
        for traj in sample.trajectories[:max_conditions]:
            all_times.extend(traj['t'].tolist())
            for conc in traj['concentrations'].values():
                all_conc.extend(conc.tolist())
    conc_mean = np.mean(all_conc)
    conc_std = np.std(all_conc) + 1e-8
    time_max = np.max(all_times) + 1e-8

    for sample in samples:
        trajectories = []
        conditions_list = []

        for traj_data in sample.trajectories[:max_conditions]:
            concentrations = traj_data['concentrations']
            t = traj_data['t']
            conditions = traj_data['conditions']

            if 'S' in concentrations:
                S = concentrations['S']
            elif 'A' in concentrations:
                S = concentrations['A']
            else:
                S = list(concentrations.values())[0]

            if len(t) != n_timepoints:
                t_new = np.linspace(t[0], t[-1], n_timepoints)
                S = np.interp(t_new, t, S)
                t = t_new

            t_norm = t / time_max
            S_norm = (S - conc_mean) / conc_std
            traj_features = np.stack([t_norm, S_norm], axis=-1)
            trajectories.append(traj_features)

            # 6 condition features
            cond_features = np.zeros(6)
            S0 = conditions.get('S0', conditions.get('A0', 1.0))
            cond_features[0] = np.log10(max(S0, 1e-9))
            I0 = conditions.get('I0', conditions.get('B0', 0))
            cond_features[1] = np.log10(max(I0, 1e-9)) if I0 > 0 else -9.0
            cond_features[2] = np.log10(max(conditions.get('E0', 1e-3), 1e-12))
            cond_features[3] = (conditions.get('T', 298.15) - 298.15) / 20.0
            cond_features[4] = (conditions.get('pH', 7.0) - 7.0) / 2.0
            P0 = conditions.get('P0', 0)
            cond_features[5] = np.log10(max(P0, 1e-9)) if P0 > 0 else -9.0
            conditions_list.append(cond_features)

        # Pad
        n_actual = len(trajectories)
        n_pad = max_conditions - n_actual
        if n_pad > 0:
            for _ in range(n_pad):
                trajectories.append(np.zeros((n_timepoints, 2)))
                conditions_list.append(np.zeros(6))

        condition_mask = np.array([False] * n_actual + [True] * n_pad)

        traj_tensor = torch.tensor(np.stack(trajectories), dtype=torch.float32).unsqueeze(0).to(device)
        cond_tensor = torch.tensor(np.stack(conditions_list), dtype=torch.float32).unsqueeze(0).to(device)
        mask_tensor = torch.tensor(condition_mask, dtype=torch.bool).unsqueeze(0).to(device)

        output = model(traj_tensor, cond_tensor, condition_mask=mask_tensor)
        pred_idx = output['logits'].argmax(dim=-1).item()
        true_idx = sample.mechanism_idx

        true_mech = MECHANISMS[true_idx]
        per_mechanism[true_mech]['total'] += 1
        if pred_idx == true_idx:
            correct += 1
            per_mechanism[true_mech]['correct'] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    per_mech_acc = {
        m: per_mechanism[m]['correct'] / per_mechanism[m]['total']
        if per_mechanism[m]['total'] > 0 else 0
        for m in MECHANISMS
    }

    return {'accuracy': accuracy, 'per_mechanism': per_mech_acc}


@torch.no_grad()
def evaluate_model(model, samples: list, device: torch.device) -> dict:
    """Evaluate a model on samples."""
    dataset = MultiConditionDataset(samples)

    correct = 0
    total = 0
    per_mechanism = {m: {'correct': 0, 'total': 0} for m in MECHANISMS}

    for i in range(len(dataset)):
        batch = dataset[i]

        trajectories = batch['trajectories'].unsqueeze(0).to(device)
        conditions = batch['conditions'].unsqueeze(0).to(device)
        derived_features = batch['derived_features'].unsqueeze(0).to(device)
        condition_mask = batch['condition_mask'].unsqueeze(0).to(device)

        output = model(
            trajectories, conditions,
            derived_features=derived_features,
            condition_mask=condition_mask,
        )

        pred_idx = output['logits'].argmax(dim=-1).item()
        true_idx = batch['mechanism_idx'].item()

        true_mech = MECHANISMS[true_idx]

        per_mechanism[true_mech]['total'] += 1
        if pred_idx == true_idx:
            correct += 1
            per_mechanism[true_mech]['correct'] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    per_mech_acc = {
        m: per_mechanism[m]['correct'] / per_mechanism[m]['total']
        if per_mechanism[m]['total'] > 0 else 0
        for m in MECHANISMS
    }

    return {'accuracy': accuracy, 'per_mechanism': per_mech_acc}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-test-samples', type=int, default=100,
                       help='Test samples per mechanism')
    parser.add_argument('--skip-classical', action='store_true',
                       help='Skip classical baseline (slow)')
    parser.add_argument('--test-set', type=str, default=None,
                       help='Path to pre-generated test set')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("TACTIC All Versions Comparison")
    print("="*70)
    print(f"Device: {device}")

    # Generate or load test set
    if args.test_set:
        print(f"\nLoading test set: {args.test_set}")
        samples, _ = load_dataset(args.test_set)
    else:
        print(f"\nGenerating test set ({args.n_test_samples} per mechanism)...")
        config = MultiConditionConfig(n_conditions_per_sample=20)
        generator = MultiConditionGenerator(config, seed=99999)
        samples = generator.generate_batch(args.n_test_samples, n_workers=1)

    print(f"Test samples: {len(samples)}")

    results = {}

    # ========== v0 ==========
    v0_checkpoint = base_dir / "checkpoints/v0/best_model.pt"
    if v0_checkpoint.exists():
        print("\n--- Evaluating v0 (single-curve) ---")
        model = load_model(v0_checkpoint, device, version='v0')
        results['v0'] = evaluate_v0(model, samples, device)
        print(f"v0 accuracy: {results['v0']['accuracy']*100:.1f}%")
    else:
        print("\nv0 checkpoint not found - skipping")

    # ========== v1 ==========
    v1_checkpoint = base_dir / "checkpoints/v1/best_model.pt"
    if v1_checkpoint.exists():
        print("\n--- Evaluating v1 (basic multi-condition) ---")
        model = load_model(v1_checkpoint, device, version='v1')
        results['v1'] = evaluate_v1(model, samples, device)
        print(f"v1 accuracy: {results['v1']['accuracy']*100:.1f}%")
    else:
        print("\nv1 checkpoint not found - skipping")

    # ========== v2 ==========
    v2_checkpoint = base_dir / "checkpoints/v2/best_model.pt"
    if v2_checkpoint.exists():
        print("\n--- Evaluating v2 (improved multi-condition) ---")
        model = load_model(v2_checkpoint, device, version='v2')
        results['v2'] = evaluate_model(model, samples, device)
        print(f"v2 accuracy: {results['v2']['accuracy']*100:.1f}%")
    else:
        print("\nv2 checkpoint not found - skipping")

    # ========== v3 ==========
    v3_checkpoint = base_dir / "checkpoints/v3/best_model.pt"
    if v3_checkpoint.exists():
        print("\n--- Evaluating v3 (multi-task) ---")
        model = load_model(v3_checkpoint, device, version='v3')
        results['v3'] = evaluate_model(model, samples, device)
        print(f"v3 accuracy: {results['v3']['accuracy']*100:.1f}%")
    else:
        print("\nv3 checkpoint not found - skipping")

    # ========== Classical ==========
    if not args.skip_classical:
        print("\n--- Evaluating Classical (AIC) ---")
        classical = evaluate_classical_baseline(samples, criterion='aic')
        results['classical'] = {
            'accuracy': classical['accuracy'],
            'per_mechanism': classical['per_mechanism'],
        }
        print(f"Classical accuracy: {results['classical']['accuracy']*100:.1f}%")

    # ========== Summary ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Method':<20} {'Accuracy':>10}")
    print("-"*35)
    print(f"{'Random baseline':<20} {'10.0%':>10}")

    for version in ['classical', 'v0', 'v1', 'v2', 'v3']:
        if version in results:
            acc = results[version]['accuracy'] * 100
            print(f"{version:<20} {acc:>9.1f}%")

    # Per-mechanism table
    print("\n" + "-"*70)
    print("Per-Mechanism Accuracy")
    print("-"*70)

    header = f"{'Mechanism':<35}"
    for v in ['v0', 'v1', 'v2', 'v3', 'classical']:
        if v in results:
            header += f" {v:>8}"
    print(header)
    print("-"*70)

    for mech in MECHANISMS:
        row = f"{mech:<35}"
        for v in ['v0', 'v1', 'v2', 'v3', 'classical']:
            if v in results:
                acc = results[v]['per_mechanism'].get(mech, 0) * 100
                row += f" {acc:>7.1f}%"
        print(row)

    # Save results
    output_path = base_dir / "results" / f"all_versions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    return results


if __name__ == "__main__":
    main()
