"""
Multi-condition classifier for TACTIC-Kinetics.

Key insight: mechanism discrimination requires COMPARING how kinetics
change across conditions. The cross-attention learns these comparisons.

For example:
- Competitive: As [I] increases, Km_app increases but Vmax unchanged
- Uncompetitive: As [I] increases, both Km_app and Vmax decrease proportionally
- Mixed: Different shift pattern
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1)]


class TrajectoryEncoder(nn.Module):
    """
    Encodes a single trajectory using 1D convolutions + transformer.

    Input features per timepoint:
    - Normalized time (t / t_max)
    - Normalized concentration (conc / conc_max)
    - Rate of change (d_conc / dt, normalized)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # Input: (time_norm, conc_norm, rate_norm) = 3 features
        self.input_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Temporal convolutions to capture local patterns
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Transformer for global patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, trajectory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            trajectory: (batch, n_timepoints, 3) - [t_norm, conc_norm, rate_norm]
            mask: (batch, n_timepoints) - True for valid points

        Returns:
            h: (batch, d_model)
        """
        batch_size = trajectory.shape[0]

        # Project input
        x = self.input_proj(trajectory)  # (batch, n_timepoints, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Convolutions (need channels-first)
        x_conv = x.transpose(1, 2)  # (batch, d_model, n_timepoints)
        x_conv = self.conv_layers(x_conv)
        x = x_conv.transpose(1, 2)  # (batch, n_timepoints, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+n_timepoints, d_model)

        # Update mask for CLS token
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Return CLS token representation
        return x[:, 0]


class ConditionEncoder(nn.Module):
    """
    Encodes experimental conditions.

    Conditions include:
    - S0, I0, A0, B0, P0: concentrations (log-normalized)
    - E0: enzyme concentration
    - T: temperature
    - pH: pH
    """

    def __init__(self, n_condition_features: int = 8, d_model: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_condition_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conditions: (batch, n_condition_features)

        Returns:
            h: (batch, d_model)
        """
        return self.encoder(conditions)


class MultiConditionClassifier(nn.Module):
    """
    Main model: classifies mechanism from a SET of trajectories.

    Key insight: mechanism discrimination requires COMPARING how
    kinetics change across conditions. The cross-attention learns
    these comparisons.

    Architecture:
    1. Encode each trajectory independently
    2. Encode conditions
    3. Combine trajectory + condition representations
    4. Cross-attention across conditions (learns comparison patterns)
    5. Attention pooling over conditions
    6. Classification head
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_traj_layers: int = 2,
        n_cross_layers: int = 3,
        n_mechanisms: int = 10,
        dropout: float = 0.1,
        n_condition_features: int = 8,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_mechanisms = n_mechanisms

        # Encode individual trajectories
        self.trajectory_encoder = TrajectoryEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_traj_layers,
            dropout=dropout,
        )

        # Encode conditions
        self.condition_encoder = ConditionEncoder(
            n_condition_features=n_condition_features,
            d_model=d_model,
        )

        # Combine trajectory + condition
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention across conditions
        # THIS IS WHERE THE MODEL LEARNS:
        # "When [I] increases, Km_app increases but Vmax stays same -> competitive"
        # "When [I] increases, Km_app stays same but Vmax decreases -> uncompetitive"
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cross_attention = nn.TransformerEncoder(cross_layer, num_layers=n_cross_layers)

        # Attention pooling over conditions
        self.attention_pool = nn.Linear(d_model, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_mechanisms),
        )

    def forward(
        self,
        trajectories: torch.Tensor,
        conditions: torch.Tensor,
        traj_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            trajectories: (batch, n_conditions, n_timepoints, 3)
                Features: [t_norm, conc_norm, rate_norm]
            conditions: (batch, n_conditions, n_condition_features)
            traj_mask: (batch, n_conditions, n_timepoints) - True for INVALID points
            condition_mask: (batch, n_conditions) - True for INVALID conditions

        Returns:
            Dict with 'logits' and 'attention_weights'
        """
        batch_size, n_conditions, n_timepoints, _ = trajectories.shape

        # Encode each trajectory independently
        # Flatten batch and conditions
        traj_flat = trajectories.view(batch_size * n_conditions, n_timepoints, -1)

        if traj_mask is not None:
            traj_mask_flat = traj_mask.view(batch_size * n_conditions, n_timepoints)
        else:
            traj_mask_flat = None

        h_traj = self.trajectory_encoder(traj_flat, traj_mask_flat)
        h_traj = h_traj.view(batch_size, n_conditions, -1)  # (batch, n_cond, d_model)

        # Encode conditions
        cond_flat = conditions.view(batch_size * n_conditions, -1)
        h_cond = self.condition_encoder(cond_flat)
        h_cond = h_cond.view(batch_size, n_conditions, -1)  # (batch, n_cond, d_model)

        # Combine trajectory and condition representations
        h_combined = self.combine(torch.cat([h_traj, h_cond], dim=-1))

        # Cross-attention: each condition attends to all others
        # This learns patterns like "rate at high [I] vs rate at low [I]"
        h_cross = self.cross_attention(h_combined, src_key_padding_mask=condition_mask)

        # Attention pooling over conditions
        attn_scores = self.attention_pool(h_cross).squeeze(-1)  # (batch, n_cond)

        if condition_mask is not None:
            attn_scores = attn_scores.masked_fill(condition_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_cond)

        # Weighted sum
        h_pooled = torch.einsum('bc,bcd->bd', attn_weights, h_cross)  # (batch, d_model)

        # Classify
        logits = self.classifier(h_pooled)

        return {
            'logits': logits,
            'attention_weights': attn_weights,
        }


def create_multi_condition_model(
    d_model: int = 128,
    n_heads: int = 4,
    n_traj_layers: int = 2,
    n_cross_layers: int = 3,
    n_mechanisms: int = 10,
    dropout: float = 0.1,
) -> MultiConditionClassifier:
    """Factory function to create a multi-condition classifier."""
    return MultiConditionClassifier(
        d_model=d_model,
        n_heads=n_heads,
        n_traj_layers=n_traj_layers,
        n_cross_layers=n_cross_layers,
        n_mechanisms=n_mechanisms,
        dropout=dropout,
    )
