"""
Benchmark experiments for TACTIC-Kinetics.

This module provides benchmark experiments to evaluate model performance:
1. Mechanism Classification - Classify mechanisms from trajectory data
2. Energy Profile Reconstruction - Reconstruct energy landscapes
3. Transfer Learning - Evaluate on real kinetics data
4. Model Comparison - Compare against baselines
5. Ablation Studies - Analyze component contributions
"""

from .mechanism_classification import MechanismClassificationBenchmark
from .energy_reconstruction import EnergyReconstructionBenchmark
from .model_comparison import ModelComparisonBenchmark

__all__ = [
    "MechanismClassificationBenchmark",
    "EnergyReconstructionBenchmark",
    "ModelComparisonBenchmark",
]
