"""
Configuration presets for TACTIC-Kinetics.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for TACTIC model architecture."""

    # Encoder
    d_model: int = 256
    n_encoder_layers: int = 6
    n_encoder_heads: int = 8
    d_ff: int = 1024

    # Decoder
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # Classifier
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # General
    dropout: float = 0.1
    n_conditions: int = 4
    condition_names: Optional[List[str]] = None

    # ODE
    temperature: float = 298.15
    use_adjoint: bool = True
    ode_solver: str = "dopri5"


@dataclass
class SmallModelConfig(ModelConfig):
    """Small model for quick experiments."""
    d_model: int = 128
    n_encoder_layers: int = 3
    n_encoder_heads: int = 4
    d_ff: int = 512
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class LargeModelConfig(ModelConfig):
    """Large model for maximum performance."""
    d_model: int = 512
    n_encoder_layers: int = 8
    n_encoder_heads: int = 8
    d_ff: int = 2048
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])


# Presets
MODEL_PRESETS = {
    "small": SmallModelConfig,
    "base": ModelConfig,
    "large": LargeModelConfig,
}


def get_model_config(preset: str = "base") -> ModelConfig:
    """Get a model configuration preset."""
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[preset]()
