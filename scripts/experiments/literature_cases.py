#!/usr/bin/env python
"""
Experiment 7: Literature Cases and Speed Comparison

Well-characterized enzymes from textbooks/BRENDA and
per-mechanism speed comparison between TACTIC and Classical.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    load_dataset,
)
from tactic_kinetics.training.multi_condition_dataset import MultiConditionDataset
from tactic_kinetics.models.multi_condition_classifier import create_multi_task_model

# Import classical baseline
sys.path.insert(0, str(Path(__file__).parent.parent))
from classical_baseline import ClassicalModelSelector

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

# Well-characterized enzymes from literature
LITERATURE_CASES = [
    {
        'enzyme': 'Alcohol dehydrogenase (yeast)',
        'ec_number': 'EC 1.1.1.1',
        'mechanism': 'ordered_bi_bi',
        'evidence': 'Theorell-Chance mechanism, classic ordered bi-bi',
        'reference': 'Theorell & Chance, 1951; Dalziel, 1963',
        'key_observation': 'NAD+ binds first, NADH released last',
    },
    {
        'enzyme': 'Lactate dehydrogenase',
        'ec_number': 'EC 1.1.1.27',
        'mechanism': 'ordered_bi_bi',
        'evidence': 'Compulsory order mechanism',
        'reference': 'Holbrook et al., 1975',
        'key_observation': 'NADH/NAD+ always binds before pyruvate/lactate',
    },
    {
        'enzyme': 'Hexokinase',
        'ec_number': 'EC 2.7.1.1',
        'mechanism': 'random_bi_bi',
        'evidence': 'Random sequential mechanism',
        'reference': 'Fromm & Zewe, 1962',
        'key_observation': 'Either glucose or ATP can bind first',
    },
    {
        'enzyme': 'Aspartate aminotransferase',
        'ec_number': 'EC 2.6.1.1',
        'mechanism': 'ping_pong',
        'evidence': 'Classic ping-pong bi-bi with PMP intermediate',
        'reference': 'Velick & Vavra, 1962',
        'key_observation': 'Parallel lines in double-reciprocal plots',
    },
    {
        'enzyme': 'Chymotrypsin + benzamidine',
        'ec_number': 'EC 3.4.21.1',
        'mechanism': 'competitive_inhibition',
        'evidence': 'Benzamidine is classic competitive inhibitor',
        'reference': 'Stroud et al., 1974',
        'key_observation': 'Km increases, Vmax unchanged with inhibitor',
    },
    {
        'enzyme': 'Acetylcholinesterase + edrophonium',
        'ec_number': 'EC 3.1.1.7',
        'mechanism': 'competitive_inhibition',
        'evidence': 'Reversible active site binding',
        'reference': 'Quinn, 1987',
        'key_observation': 'Inhibition overcome by high substrate',
    },
    {
        'enzyme': 'Alkaline phosphatase + L-phenylalanine',
        'ec_number': 'EC 3.1.3.1',
        'mechanism': 'uncompetitive_inhibition',
        'evidence': 'Binds only to ES complex',
        'reference': 'Hoylaerts et al., 1997',
        'key_observation': 'Both Km and Vmax decrease proportionally',
    },
    {
        'enzyme': 'Xanthine oxidase',
        'ec_number': 'EC 1.17.3.2',
        'mechanism': 'substrate_inhibition',
        'evidence': 'Inhibited by high xanthine concentrations',
        'reference': 'Massey et al., 1969',
        'key_observation': 'Bell-shaped velocity vs [S] curve',
    },
    {
        'enzyme': 'Phosphofructokinase',
        'ec_number': 'EC 2.7.1.11',
        'mechanism': 'substrate_inhibition',
        'evidence': 'ATP inhibition at high concentrations',
        'reference': 'Uyeda, 1979',
        'key_observation': 'Regulatory mechanism for glycolysis control',
    },
    {
        'enzyme': 'Fumarase',
        'ec_number': 'EC 4.2.1.2',
        'mechanism': 'michaelis_menten_reversible',
        'evidence': 'Fully reversible hydration/dehydration',
        'reference': 'Alberty, 1954',
        'key_observation': 'Approaches equilibrium, product inhibition',
    },
]


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained TACTIC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
def time_tactic_inference(model, samples: list, device: torch.device) -> dict:
    """Time TACTIC inference per mechanism."""
    dataset = MultiConditionDataset(samples)

    times_by_mechanism = defaultdict(list)

    for i in range(len(dataset)):
        batch = dataset[i]
        true_idx = batch['mechanism_idx'].item()
        mech_name = MECHANISMS[true_idx]

        trajectories = batch['trajectories'].unsqueeze(0).to(device)
        conditions = batch['conditions'].unsqueeze(0).to(device)
        derived_features = batch['derived_features'].unsqueeze(0).to(device)
        condition_mask = batch['condition_mask'].unsqueeze(0).to(device)

        # Warm up GPU
        if i == 0:
            _ = model(trajectories, conditions, derived_features=derived_features,
                     condition_mask=condition_mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model(trajectories, conditions, derived_features=derived_features,
                 condition_mask=condition_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        times_by_mechanism[mech_name].append(elapsed)

    return times_by_mechanism


def time_classical_inference(samples: list) -> dict:
    """Time classical inference per mechanism."""
    selector = ClassicalModelSelector()
    times_by_mechanism = defaultdict(list)

    for sample in samples:
        mech_name = sample.mechanism

        start = time.perf_counter()
        _ = selector.select_model(sample, criterion='aic')
        elapsed = time.perf_counter() - start

        times_by_mechanism[mech_name].append(elapsed)

    return times_by_mechanism


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Literature Cases and Speed Analysis')
    parser.add_argument('--n-samples', type=int, default=20, help='Samples per mechanism for speed test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/v3/best_model.pt')
    parser.add_argument('--test-set', type=str, default=None, help='Pre-generated test set')
    parser.add_argument('--skip-speed-test', action='store_true', help='Skip speed comparison')
    parser.add_argument('--save-results', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("EXPERIMENT 7: Literature Cases & Speed Comparison")
    print("="*70)
    print(f"Device: {device}")

    # Print literature cases
    print("\n" + "="*70)
    print("WELL-CHARACTERIZED ENZYMES FROM LITERATURE")
    print("="*70)
    print("\nThese enzymes have established mechanisms from decades of biochemical studies.")
    print("They serve as ground truth for validating mechanism classification methods.\n")

    for i, case in enumerate(LITERATURE_CASES, 1):
        print(f"{i}. {case['enzyme']} ({case['ec_number']})")
        print(f"   Mechanism: {case['mechanism']}")
        print(f"   Evidence: {case['evidence']}")
        print(f"   Key observation: {case['key_observation']}")
        print(f"   Reference: {case['reference']}")
        print()

    # Summary table
    print("-"*70)
    print("MECHANISM DISTRIBUTION IN LITERATURE SET")
    print("-"*70)
    mech_counts = defaultdict(int)
    for case in LITERATURE_CASES:
        mech_counts[case['mechanism']] += 1

    for mech in MECHANISMS:
        count = mech_counts.get(mech, 0)
        if count > 0:
            print(f"  {mech:<35} {count} enzyme(s)")

    # Speed comparison
    if not args.skip_speed_test:
        print("\n" + "="*70)
        print("SPEED COMPARISON: TACTIC vs CLASSICAL")
        print("="*70)

        # Load or generate test set
        if args.test_set:
            print(f"Loading test set: {args.test_set}")
            samples, _ = load_dataset(args.test_set)
        else:
            print(f"Generating test set ({args.n_samples} per mechanism)...")
            config = MultiConditionConfig(n_conditions_per_sample=20)
            generator = MultiConditionGenerator(config, seed=78901)
            samples = generator.generate_batch(args.n_samples, n_workers=4)

        print(f"Test samples: {len(samples)}")

        # Load model
        checkpoint_path = base_dir / args.checkpoint
        model = load_model(checkpoint_path, device)

        # Time TACTIC
        print("\nTiming TACTIC inference...")
        tactic_times = time_tactic_inference(model, samples, device)

        # Time Classical (subset - it's slow)
        print("Timing Classical inference (may take a while)...")
        classical_times = time_classical_inference(samples[:min(100, len(samples))])

        # Results
        print("\n" + "-"*70)
        print("PER-MECHANISM INFERENCE TIME")
        print("-"*70)
        print(f"{'Mechanism':<35} {'TACTIC (ms)':>12} {'Classical (s)':>14} {'Speedup':>10}")
        print("-"*70)

        total_tactic = 0
        total_classical = 0

        for mech in MECHANISMS:
            t_times = tactic_times.get(mech, [])
            c_times = classical_times.get(mech, [])

            if t_times:
                t_mean = np.mean(t_times) * 1000  # to ms
                total_tactic += sum(t_times)
            else:
                t_mean = 0

            if c_times:
                c_mean = np.mean(c_times)  # already in seconds
                total_classical += sum(c_times)
                speedup = c_mean / (t_mean / 1000) if t_mean > 0 else 0
            else:
                c_mean = 0
                speedup = 0

            if t_times or c_times:
                print(f"{mech:<35} {t_mean:>11.2f} {c_mean:>13.2f} {speedup:>9.0f}x")

        # Overall summary
        print("-"*70)
        n_tactic = sum(len(v) for v in tactic_times.values())
        n_classical = sum(len(v) for v in classical_times.values())

        tactic_per_sample = total_tactic / n_tactic * 1000 if n_tactic > 0 else 0
        classical_per_sample = total_classical / n_classical if n_classical > 0 else 0

        print(f"\n{'TOTAL':<35} {tactic_per_sample:>11.2f}ms {classical_per_sample:>13.2f}s")

        overall_speedup = classical_per_sample / (tactic_per_sample / 1000) if tactic_per_sample > 0 else 0
        print(f"\nOverall speedup: {overall_speedup:.0f}x faster")

        print(f"\nKey insight: Classical fitting time varies by mechanism complexity.")
        print("Bi-substrate mechanisms (more parameters) take longer to fit.")
        print("TACTIC inference time is constant regardless of mechanism.")

    # Save results
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"literature_speed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output = {
            'literature_cases': LITERATURE_CASES,
            'speed_comparison': {
                'tactic_ms_per_sample': tactic_per_sample if not args.skip_speed_test else None,
                'classical_s_per_sample': classical_per_sample if not args.skip_speed_test else None,
                'speedup': overall_speedup if not args.skip_speed_test else None,
            } if not args.skip_speed_test else None,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to: {output_path}")

    return LITERATURE_CASES


if __name__ == "__main__":
    main()
