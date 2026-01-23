#!/usr/bin/env python
"""
Run all TACTIC experiments for paper.

Usage:
    python scripts/experiments/run_all_experiments.py --quick     # Fast test run
    python scripts/experiments/run_all_experiments.py --full      # Full paper-quality
    python scripts/experiments/run_all_experiments.py --exp 1 3 5 # Run specific experiments
"""

import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    1: {
        'name': 'Confidence-Accuracy Analysis',
        'script': 'confidence_analysis.py',
        'quick_args': '--n-samples 20 --save-results',
        'full_args': '--n-samples 100 --save-results',
        'time_estimate': '5 min',
    },
    2: {
        'name': 'Condition Ablation Study',
        'script': 'condition_ablation.py',
        'quick_args': '--n-samples 10 --skip-classical --save-results',
        'full_args': '--n-samples 50 --save-results',
        'time_estimate': '30 min (with classical)',
    },
    3: {
        'name': 'Noise Robustness Analysis',
        'script': 'noise_robustness.py',
        'quick_args': '--n-samples 10 --skip-classical --save-results',
        'full_args': '--n-samples 50 --save-results',
        'time_estimate': '30 min (with classical)',
    },
    4: {
        'name': 'Family-Level Accuracy',
        'script': 'family_accuracy.py',
        'quick_args': '--n-samples 20 --skip-classical --save-results',
        'full_args': '--n-samples 100 --save-results',
        'time_estimate': '10 min',
    },
    5: {
        'name': 'Identifiability Analysis',
        'script': 'identifiability_analysis.py',
        'quick_args': '--n-samples 20 --save-results',
        'full_args': '--n-samples 100 --save-results',
        'time_estimate': '5 min',
    },
    6: {
        'name': 'Error Correlation Analysis',
        'script': 'error_correlation.py',
        'quick_args': '--n-samples 10 --save-results',
        'full_args': '--n-samples 50 --save-results',
        'time_estimate': '30 min (classical is slow)',
    },
    7: {
        'name': 'Literature Cases & Speed',
        'script': 'literature_cases.py',
        'quick_args': '--n-samples 5 --save-results',
        'full_args': '--n-samples 20 --save-results',
        'time_estimate': '15 min',
    },
}


def run_experiment(exp_id: int, mode: str = 'quick'):
    """Run a single experiment."""
    exp = EXPERIMENTS[exp_id]
    script_dir = Path(__file__).parent
    script_path = script_dir / exp['script']

    args = exp['quick_args'] if mode == 'quick' else exp['full_args']

    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_id}: {exp['name']}")
    print(f"{'='*70}")
    print(f"Script: {exp['script']}")
    print(f"Args: {args}")
    print(f"Estimated time: {exp['time_estimate']}")
    print("-"*70)

    cmd = [sys.executable, str(script_path)] + args.split()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Experiment {exp_id} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment {exp_id} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Experiment {exp_id} failed: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run TACTIC experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test run (small samples)')
    parser.add_argument('--full', action='store_true', help='Full paper-quality run')
    parser.add_argument('--exp', type=int, nargs='+', help='Run specific experiments (1-7)')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    args = parser.parse_args()

    if args.list:
        print("="*70)
        print("TACTIC EXPERIMENTS")
        print("="*70)
        for exp_id, exp in EXPERIMENTS.items():
            print(f"\n{exp_id}. {exp['name']}")
            print(f"   Script: {exp['script']}")
            print(f"   Time: {exp['time_estimate']}")
        return

    mode = 'full' if args.full else 'quick'
    exp_ids = args.exp if args.exp else list(EXPERIMENTS.keys())

    print("="*70)
    print(f"RUNNING TACTIC EXPERIMENTS ({mode.upper()} mode)")
    print("="*70)
    print(f"Experiments to run: {exp_ids}")

    results = {}
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_id}")
            continue
        results[exp_id] = run_experiment(exp_id, mode)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for exp_id, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} Experiment {exp_id}: {EXPERIMENTS[exp_id]['name']}")

    n_success = sum(results.values())
    n_total = len(results)
    print(f"\nCompleted: {n_success}/{n_total}")

    if n_success == n_total:
        print("\nAll experiments completed! Results saved to results/experiments/")


if __name__ == "__main__":
    main()
