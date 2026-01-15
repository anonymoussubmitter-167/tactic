"""
Observation encoder for TACTIC-Kinetics.

This module implements a Transformer-based encoder that takes sparse
progress curve observations and experimental conditions as input and
produces a latent representation for mechanism classification and
energy prediction.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple


class TimeEmbedding(nn.Module):
    """
    Learnable time embedding using sinusoidal positional encoding.

    Maps scalar time values to high-dimensional embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_time: float = 1000.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time

        if learnable:
            # Learnable frequency components
            self.freq = nn.Parameter(torch.randn(d_model // 2) * 0.01)
            self.phase = nn.Parameter(torch.zeros(d_model // 2))
        else:
            # Fixed sinusoidal frequencies
            freq = torch.exp(
                torch.arange(0, d_model // 2) * (-math.log(max_time) / (d_model // 2 - 1))
            )
            self.register_buffer("freq", freq)
            self.register_buffer("phase", torch.zeros(d_model // 2))

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, t: Tensor) -> Tensor:
        """
        Embed time values.

        Args:
            t: Time values, shape (batch, n_times) or (batch, n_times, 1)

        Returns:
            Time embeddings, shape (batch, n_times, d_model)
        """
        if t.dim() == 3:
            t = t.squeeze(-1)

        # Compute sinusoidal embeddings
        # t: (batch, n_times), freq: (d_model // 2)
        t = t.unsqueeze(-1)  # (batch, n_times, 1)
        freq = self.freq.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model // 2)
        phase = self.phase.unsqueeze(0).unsqueeze(0)

        angles = t * freq + phase
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)

        # Concatenate sin and cos
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (batch, n_times, d_model)

        return self.linear(emb)


class ConditionEmbedding(nn.Module):
    """
    Embedding for experimental conditions.

    Maps continuous condition values (temperature, pH, concentrations, etc.)
    to embeddings.
    """

    def __init__(
        self,
        n_conditions: int,
        d_model: int,
        condition_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        self.d_model = d_model
        self.condition_names = condition_names or [f"cond_{i}" for i in range(n_conditions)]

        # Linear projection for each condition
        self.embeddings = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for name in self.condition_names
        })

        # Combine all conditions
        self.combiner = nn.Sequential(
            nn.Linear(d_model * n_conditions, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, conditions: Tensor) -> Tensor:
        """
        Embed experimental conditions.

        Args:
            conditions: Condition values, shape (batch, n_conditions)

        Returns:
            Condition embedding, shape (batch, d_model)
        """
        embeddings = []
        for i, name in enumerate(self.condition_names):
            cond_val = conditions[:, i:i+1]  # (batch, 1)
            emb = self.embeddings[name](cond_val)  # (batch, d_model)
            embeddings.append(emb)

        combined = torch.cat(embeddings, dim=-1)  # (batch, n_conditions * d_model)
        return self.combiner(combined)


class ObservationTokenizer(nn.Module):
    """
    Converts observation (time, value) pairs into tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_observables: int = 1,
        max_time: float = 1000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_observables = n_observables

        # Time embedding
        self.time_embed = TimeEmbedding(d_model, max_time)

        # Value embedding (linear projection)
        self.value_embed = nn.Sequential(
            nn.Linear(n_observables, d_model),
            nn.LayerNorm(d_model),
        )

        # Combine time and value embeddings
        self.combiner = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        times: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Tokenize observations.

        Args:
            times: Time points, shape (batch, n_obs)
            values: Observed values, shape (batch, n_obs) or (batch, n_obs, n_observables)
            mask: Optional mask for valid observations, shape (batch, n_obs)

        Returns:
            Tuple of (tokens, mask):
                tokens: shape (batch, n_obs, d_model)
                mask: shape (batch, n_obs), True for valid positions
        """
        if values.dim() == 2:
            values = values.unsqueeze(-1)  # (batch, n_obs, 1)

        time_emb = self.time_embed(times)  # (batch, n_obs, d_model)
        value_emb = self.value_embed(values)  # (batch, n_obs, d_model)

        combined = torch.cat([time_emb, value_emb], dim=-1)
        tokens = self.combiner(combined)

        if mask is None:
            mask = torch.ones(times.shape[:2], dtype=torch.bool, device=times.device)

        return tokens, mask


class ObservationEncoder(nn.Module):
    """
    Transformer encoder for sparse observations.

    Takes observation tokens and experimental conditions, and produces
    a latent representation suitable for mechanism classification and
    energy prediction.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_observables: int = 1,
        n_conditions: int = 4,
        max_time: float = 1000.0,
        condition_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Tokenization
        self.tokenizer = ObservationTokenizer(
            d_model=d_model,
            n_observables=n_observables,
            max_time=max_time,
        )

        # Condition embedding
        if n_conditions > 0:
            self.condition_embed = ConditionEmbedding(
                n_conditions=n_conditions,
                d_model=d_model,
                condition_names=condition_names,
            )
        else:
            self.condition_embed = None

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cond_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        times: Tensor,
        values: Tensor,
        conditions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode observations to latent representation.

        Args:
            times: Observation times, shape (batch, n_obs)
            values: Observed values, shape (batch, n_obs) or (batch, n_obs, n_observables)
            conditions: Experimental conditions, shape (batch, n_conditions)
            mask: Valid observation mask, shape (batch, n_obs)

        Returns:
            Latent representation, shape (batch, d_model)
        """
        batch_size = times.shape[0]

        # Tokenize observations
        obs_tokens, obs_mask = self.tokenizer(times, values, mask)

        # Build sequence: [CLS] [COND] [OBS_1] [OBS_2] ...
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        mask_list = [torch.ones(batch_size, 1, dtype=torch.bool, device=times.device)]

        if self.condition_embed is not None and conditions is not None:
            cond_emb = self.condition_embed(conditions)  # (batch, d_model)
            cond_token = self.cond_token + cond_emb.unsqueeze(1)
            tokens.append(cond_token)
            mask_list.append(torch.ones(batch_size, 1, dtype=torch.bool, device=times.device))

        tokens.append(obs_tokens)
        mask_list.append(obs_mask)

        # Concatenate
        tokens = torch.cat(tokens, dim=1)  # (batch, seq_len, d_model)
        mask = torch.cat(mask_list, dim=1)  # (batch, seq_len)

        # Create attention mask (True = masked out)
        attn_mask = ~mask

        # Apply transformer
        encoded = self.transformer(tokens, src_key_padding_mask=attn_mask)

        # Extract CLS token output
        cls_output = encoded[:, 0]  # (batch, d_model)

        return self.output_norm(cls_output)

    def encode_with_sequence(
        self,
        times: Tensor,
        values: Tensor,
        conditions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode observations and return both CLS token and full sequence.

        Returns:
            Tuple of (cls_output, sequence_output):
                cls_output: shape (batch, d_model)
                sequence_output: shape (batch, seq_len, d_model)
        """
        batch_size = times.shape[0]

        # Tokenize observations
        obs_tokens, obs_mask = self.tokenizer(times, values, mask)

        # Build sequence
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        mask_list = [torch.ones(batch_size, 1, dtype=torch.bool, device=times.device)]

        if self.condition_embed is not None and conditions is not None:
            cond_emb = self.condition_embed(conditions)
            cond_token = self.cond_token + cond_emb.unsqueeze(1)
            tokens.append(cond_token)
            mask_list.append(torch.ones(batch_size, 1, dtype=torch.bool, device=times.device))

        tokens.append(obs_tokens)
        mask_list.append(obs_mask)

        tokens = torch.cat(tokens, dim=1)
        mask = torch.cat(mask_list, dim=1)
        attn_mask = ~mask

        encoded = self.transformer(tokens, src_key_padding_mask=attn_mask)
        cls_output = self.output_norm(encoded[:, 0])

        return cls_output, encoded


class SetEncoder(nn.Module):
    """
    Alternative encoder using Deep Sets / Set Transformer architecture.

    More suitable for sets of observations where order doesn't matter.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_observables: int = 1,
        n_conditions: int = 4,
        n_inducing: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_inducing = n_inducing

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(1 + n_observables, d_model),  # time + value
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Inducing points for Set Transformer
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Condition embedding
        if n_conditions > 0:
            self.condition_proj = nn.Linear(n_conditions, d_model)
        else:
            self.condition_proj = None

        # Output pooling
        self.output_pool = nn.Sequential(
            nn.Linear(n_inducing * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        times: Tensor,
        values: Tensor,
        conditions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode observations using Set Transformer.

        Args:
            times: shape (batch, n_obs)
            values: shape (batch, n_obs) or (batch, n_obs, n_observables)
            conditions: shape (batch, n_conditions)
            mask: shape (batch, n_obs)

        Returns:
            Latent representation, shape (batch, d_model)
        """
        batch_size = times.shape[0]

        if values.dim() == 2:
            values = values.unsqueeze(-1)

        # Combine time and value
        x = torch.cat([times.unsqueeze(-1), values], dim=-1)  # (batch, n_obs, 1 + n_obs)
        x = self.input_proj(x)  # (batch, n_obs, d_model)

        # Prepare inducing points
        h = self.inducing_points.expand(batch_size, -1, -1)  # (batch, n_inducing, d_model)

        # Cross-attention layers
        key_padding_mask = ~mask if mask is not None else None

        for i in range(len(self.cross_attn_layers)):
            # Cross-attention: inducing points attend to observations
            h_attn, _ = self.cross_attn_layers[i](h, x, x, key_padding_mask=key_padding_mask)
            h = self.norms1[i](h + h_attn)
            h = self.norms2[i](h + self.ff_layers[i](h))

        # Pool inducing points
        h_flat = h.reshape(batch_size, -1)  # (batch, n_inducing * d_model)
        output = self.output_pool(h_flat)  # (batch, d_model)

        # Add condition information
        if self.condition_proj is not None and conditions is not None:
            cond_emb = self.condition_proj(conditions)
            output = output + cond_emb

        return output
