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
    - Normalized substrate concentration (S / conc_max)
    - Normalized product concentration (P / conc_max)
    - Rate of substrate change (dS/dt, normalized)
    - Rate of product change (dP/dt, normalized)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, n_traj_features: int = 5):
        super().__init__()

        # Input: (t_norm, S_norm, P_norm, dS/dt_norm, dP/dt_norm) = 5 features
        self.input_proj = nn.Sequential(
            nn.Linear(n_traj_features, d_model),
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


class DerivedFeatureEncoder(nn.Module):
    """
    Encodes derived kinetic features (v0, t_half, rate_ratio, etc.).

    These are pre-computed features that biochemists use for mechanism discrimination.
    """

    def __init__(self, n_derived_features: int = 8, d_model: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_derived_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, derived_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            derived_features: (batch, n_derived_features)

        Returns:
            h: (batch, d_model)
        """
        return self.encoder(derived_features)


class PairwiseComparisonModule(nn.Module):
    """
    Explicitly computes pairwise differences between conditions.

    This helps the model learn "how does kinetics change when [I] increases?"
    by directly comparing pairs of conditions.

    Key insight: mechanism discrimination often comes down to comparing
    how kinetic parameters change between two conditions (e.g., high vs low [I]).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Encode pairwise trajectory differences
        self.diff_encoder = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # concat, diff, product
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Encode condition differences
        self.condition_diff_encoder = nn.Sequential(
            nn.Linear(16, d_model),  # Condition differences (8 + 8)
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Combine trajectory diff with condition diff
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Self-attention over pairs
        self.pair_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2
        )

    def forward(self, h_combined: torch.Tensor, conditions: torch.Tensor,
                condition_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h_combined: (batch, n_conditions, d_model) - combined traj+cond representations
            conditions: (batch, n_conditions, n_cond_features)
            condition_mask: (batch, n_conditions) - True for INVALID conditions

        Returns:
            h_pairwise: (batch, d_model) - aggregated pairwise comparison features
        """
        batch_size, n_cond, d_model = h_combined.shape

        # Generate all pairs (i, j) where i < j
        pairs = []
        cond_diffs = []
        pair_masks = []

        for i in range(min(n_cond, 10)):  # Limit to first 10 conditions for efficiency
            for j in range(i + 1, min(n_cond, 10)):
                # Trajectory difference features
                h_i = h_combined[:, i, :]  # (batch, d_model)
                h_j = h_combined[:, j, :]

                # Difference, element-wise product (interaction), and average
                h_diff = h_i - h_j
                h_prod = h_i * h_j
                pair_feat = torch.cat([h_diff, h_prod, (h_i + h_j) / 2], dim=-1)
                pairs.append(pair_feat)

                # Condition difference
                c_i = conditions[:, i, :]
                c_j = conditions[:, j, :]
                cond_diff = torch.cat([c_i - c_j, c_i + c_j], dim=-1)
                cond_diffs.append(cond_diff)

                # Pair mask: invalid if either condition is invalid
                if condition_mask is not None:
                    mask_i = condition_mask[:, i]
                    mask_j = condition_mask[:, j]
                    pair_mask = mask_i | mask_j
                    pair_masks.append(pair_mask)

        if len(pairs) == 0:
            return torch.zeros(batch_size, d_model, device=h_combined.device)

        # Stack pairs: (batch, n_pairs, features)
        pairs = torch.stack(pairs, dim=1)
        cond_diffs = torch.stack(cond_diffs, dim=1)

        # Encode pairs
        h_pairs = self.diff_encoder(pairs)
        h_cond_diff = self.condition_diff_encoder(cond_diffs)

        # Combine
        h_pairwise = self.combine(torch.cat([h_pairs, h_cond_diff], dim=-1))

        # Create pair mask if needed
        if len(pair_masks) > 0:
            pair_mask = torch.stack(pair_masks, dim=1)  # (batch, n_pairs)
        else:
            pair_mask = None

        # Self-attention over pairs
        h_pairwise = self.pair_attention(h_pairwise, src_key_padding_mask=pair_mask)

        # Mean pooling over pairs
        if pair_mask is not None:
            # Mask out invalid pairs for mean calculation
            mask_expanded = (~pair_mask).unsqueeze(-1).float()
            h_pooled = (h_pairwise * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            h_pooled = h_pairwise.mean(dim=1)

        return h_pooled


class MultiConditionClassifier(nn.Module):
    """
    Main model: classifies mechanism from a SET of trajectories.

    Key insight: mechanism discrimination requires COMPARING how
    kinetics change across conditions. The cross-attention learns
    these comparisons.

    Architecture:
    1. Encode each trajectory independently (with S, P, rates)
    2. Encode conditions ([S], [I], [A], [B], etc.)
    3. Encode derived kinetic features (v0, t_half, rate_ratio)
    4. Combine all representations
    5. Cross-attention across conditions (learns comparison patterns)
    6. Pairwise comparison module (explicit condition comparisons)
    7. Attention pooling + pairwise pooling
    8. Classification head
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
        n_traj_features: int = 5,
        n_derived_features: int = 8,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_mechanisms = n_mechanisms

        # Encode individual trajectories (now with S, P, and rates)
        self.trajectory_encoder = TrajectoryEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_traj_layers,
            dropout=dropout,
            n_traj_features=n_traj_features,
        )

        # Encode conditions
        self.condition_encoder = ConditionEncoder(
            n_condition_features=n_condition_features,
            d_model=d_model,
        )

        # Encode derived kinetic features
        self.derived_encoder = DerivedFeatureEncoder(
            n_derived_features=n_derived_features,
            d_model=d_model,
        )

        # Combine trajectory + condition + derived features
        self.combine = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # 3 encoders
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

        # Pairwise comparison module
        self.pairwise = PairwiseComparisonModule(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Attention pooling over conditions
        self.attention_pool = nn.Linear(d_model, 1)

        # Classification head (uses both pooled and pairwise features)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # pooled + pairwise
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_mechanisms),
        )

    def forward(
        self,
        trajectories: torch.Tensor,
        conditions: torch.Tensor,
        derived_features: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            trajectories: (batch, n_conditions, n_timepoints, 5)
                Features: [t_norm, S_norm, P_norm, dS/dt_norm, dP/dt_norm]
            conditions: (batch, n_conditions, n_condition_features)
            derived_features: (batch, n_conditions, n_derived_features)
                Pre-computed kinetic features: v0, t_half, rate_ratio, etc.
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

        # Encode derived features
        if derived_features is not None:
            derived_flat = derived_features.view(batch_size * n_conditions, -1)
            h_derived = self.derived_encoder(derived_flat)
            h_derived = h_derived.view(batch_size, n_conditions, -1)
        else:
            # If no derived features, use zeros
            h_derived = torch.zeros_like(h_traj)

        # Combine trajectory, condition, and derived representations
        h_combined = self.combine(torch.cat([h_traj, h_cond, h_derived], dim=-1))

        # Cross-attention: each condition attends to all others
        # This learns patterns like "rate at high [I] vs rate at low [I]"
        h_cross = self.cross_attention(h_combined, src_key_padding_mask=condition_mask)

        # Pairwise comparison (explicit comparison between conditions)
        h_pairwise = self.pairwise(h_combined, conditions, condition_mask)

        # Attention pooling over conditions
        attn_scores = self.attention_pool(h_cross).squeeze(-1)  # (batch, n_cond)

        if condition_mask is not None:
            attn_scores = attn_scores.masked_fill(condition_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_cond)

        # Weighted sum
        h_pooled = torch.einsum('bc,bcd->bd', attn_weights, h_cross)  # (batch, d_model)

        # Combine pooled representation with pairwise features
        h_final = torch.cat([h_pooled, h_pairwise], dim=-1)

        # Classify
        logits = self.classifier(h_final)

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
    n_condition_features: int = 8,
    n_traj_features: int = 5,
    n_derived_features: int = 8,
) -> MultiConditionClassifier:
    """Factory function to create a multi-condition classifier."""
    return MultiConditionClassifier(
        d_model=d_model,
        n_heads=n_heads,
        n_traj_layers=n_traj_layers,
        n_cross_layers=n_cross_layers,
        n_mechanisms=n_mechanisms,
        dropout=dropout,
        n_condition_features=n_condition_features,
        n_traj_features=n_traj_features,
        n_derived_features=n_derived_features,
    )
