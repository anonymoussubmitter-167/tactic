#!/usr/bin/env python
"""
Generate a fixed test set for fair comparison between methods.

This ensures TACTIC and classical methods are evaluated on exactly
the same samples, enabling valid statistical comparison.

Usage:
    # Generate small test set for quick experiments
    python scripts/generate_test_set.py --n-samples 20 --output data/test_small.pt

    # Generate paper-quality test set
    python scripts/generate_test_set.py --n-samples 200 --output data/test_paper.pt
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionGenerator,
    MultiConditionConfig,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(description='Generate fixed test set')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Samples per mechanism')
    parser.add_argument('--n-conditions', type=int, default=20,
                       help='Conditions per sample')
    parser.add_argument('--seed', type=int, default=99999,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path (.pt file)')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of CPU workers (default: all)')
    args = parser.parse_args()

    print("="*70)
    print("Generating Fixed Test Set")
    print("="*70)
    print(f"Samples per mechanism: {args.n_samples}")
    print(f"Conditions per sample: {args.n_conditions}")
    print(f"Random seed: {args.seed}")
    print(f"Output path: {args.output}")

    config = MultiConditionConfig(
        n_conditions_per_sample=args.n_conditions,
        n_timepoints=20,
        noise_level=0.03,
    )

    generator = MultiConditionGenerator(config, seed=args.seed)
    samples = generator.generate_batch(args.n_samples, n_workers=args.n_workers)

    print(f"\nGenerated {len(samples)} samples")

    # Count per mechanism
    from collections import Counter
    mech_counts = Counter(s.mechanism for s in samples)
    print("\nPer-mechanism counts:")
    for mech, count in sorted(mech_counts.items()):
        print(f"  {mech}: {count}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(samples, str(output_path), config)

    print(f"\nSaved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
