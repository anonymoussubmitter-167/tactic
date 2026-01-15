"""
Training utilities for TACTIC-Kinetics.
"""

from .trainer import TACTICTrainer, TrainingConfig
from .synthetic_data import (
    SyntheticDataGenerator,
    generate_mechanism_dataset,
)

__all__ = [
    "TACTICTrainer",
    "TrainingConfig",
    "SyntheticDataGenerator",
    "generate_mechanism_dataset",
]
