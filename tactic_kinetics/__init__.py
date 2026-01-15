"""
TACTIC-Kinetics: Thermodynamic-Native Inference for Enzyme Mechanism Discovery

A framework for enzyme kinetics inference that:
1. Parameterizes kinetics directly in Gibbs energy coordinates
2. Performs mechanism discrimination via energy profile classification
3. Learns transferable priors grounded in physical chemistry

Main components:
- models: Neural network architecture (encoder, decoder, classifier, ODE simulator)
- mechanisms: Enzyme mechanism templates and graph structures
- training: Training pipeline and synthetic data generation
- data: Data loading and eQuilibrator integration
- utils: Thermodynamic utility functions
"""

__version__ = "0.1.0"

# Core imports that don't require torchdiffeq
from .mechanisms.templates import (
    get_mechanism_by_name,
    get_all_mechanisms,
    MichaelisMentenIrreversible,
    MichaelisMentenReversible,
    CompetitiveInhibition,
    UncompetitiveInhibition,
    MixedInhibition,
    SubstrateInhibition,
    OrderedBiBi,
    RandomBiBi,
    PingPong,
)

# Lazy imports for components that require torchdiffeq
def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "TACTICKinetics":
        from .models.tactic_model import TACTICKinetics
        return TACTICKinetics
    elif name == "TACTICLoss":
        from .models.tactic_model import TACTICLoss
        return TACTICLoss
    elif name == "TACTICTrainer":
        from .training.trainer import TACTICTrainer
        return TACTICTrainer
    elif name == "TrainingConfig":
        from .training.trainer import TrainingConfig
        return TrainingConfig
    elif name == "SyntheticDataGenerator":
        from .training.synthetic_data import SyntheticDataGenerator
        return SyntheticDataGenerator
    elif name == "generate_mechanism_dataset":
        from .training.synthetic_data import generate_mechanism_dataset
        return generate_mechanism_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main model (requires torchdiffeq)
    "TACTICKinetics",
    "TACTICLoss",
    # Mechanisms (always available)
    "get_mechanism_by_name",
    "get_all_mechanisms",
    "MichaelisMentenIrreversible",
    "MichaelisMentenReversible",
    "CompetitiveInhibition",
    "UncompetitiveInhibition",
    "MixedInhibition",
    "SubstrateInhibition",
    "OrderedBiBi",
    "RandomBiBi",
    "PingPong",
    # Training (requires torchdiffeq)
    "TACTICTrainer",
    "TrainingConfig",
    "SyntheticDataGenerator",
    "generate_mechanism_dataset",
]
