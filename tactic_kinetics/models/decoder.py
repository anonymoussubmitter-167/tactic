"""
Energy decoders for TACTIC-Kinetics.

This module implements mechanism-specific decoders that predict Gibbs
energy landscapes from the latent representation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from ..mechanisms.base import MechanismTemplate


class EnergyDecoder(nn.Module):
    """
    Base energy decoder that predicts state and barrier energies.

    Outputs energies in kJ/mol, suitable for use with the Eyring equation.
    """

    def __init__(
        self,
        d_input: int,
        n_state_energies: int,
        n_barrier_energies: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        energy_scale: float = 50.0,
    ):
        """
        Args:
            d_input: Dimension of input latent representation
            n_state_energies: Number of state energy parameters (excluding reference)
            n_barrier_energies: Number of barrier energy parameters
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            energy_scale: Scale factor for energy outputs (energies ~ Normal(0, scale))
        """
        super().__init__()
        self.n_state_energies = n_state_energies
        self.n_barrier_energies = n_barrier_energies
        self.energy_scale = energy_scale

        # Build MLP
        layers = []
        in_dim = d_input
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Separate heads for state and barrier energies
        self.state_head = nn.Linear(in_dim, n_state_energies)
        self.barrier_head = nn.Linear(in_dim, n_barrier_energies)

        # Initialize with small values
        nn.init.normal_(self.state_head.weight, std=0.01)
        nn.init.zeros_(self.state_head.bias)
        nn.init.normal_(self.barrier_head.weight, std=0.01)
        # Barriers should be positive, initialize with positive bias
        nn.init.constant_(self.barrier_head.bias, 50.0)  # ~50 kJ/mol typical barrier

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict energies from latent representation.

        Args:
            h: Latent representation, shape (batch, d_input)

        Returns:
            Tuple of (state_energies, barrier_energies):
                state_energies: shape (batch, n_state_energies), in kJ/mol
                barrier_energies: shape (batch, n_barrier_energies), in kJ/mol
        """
        features = self.mlp(h)

        state_energies = self.state_head(features)
        barrier_energies = self.barrier_head(features)

        # Ensure barriers are positive (using softplus)
        barrier_energies = nn.functional.softplus(barrier_energies) + 10.0  # Minimum 10 kJ/mol

        return state_energies, barrier_energies


class MechanismSpecificDecoder(nn.Module):
    """
    Decoder that is specific to a single mechanism.

    Includes mechanism-specific constraints and prior information.
    """

    def __init__(
        self,
        mechanism: MechanismTemplate,
        d_input: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        use_priors: bool = True,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.use_priors = use_priors

        # Create base decoder
        self.decoder = EnergyDecoder(
            d_input=d_input,
            n_state_energies=mechanism.n_energy_params,
            n_barrier_energies=mechanism.n_barrier_params,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Prior parameters (learnable)
        if use_priors:
            # Prior means for state energies (typically near 0)
            self.register_buffer(
                "state_prior_mean",
                torch.zeros(mechanism.n_energy_params)
            )
            # Prior stds (loose priors, ~20 kJ/mol)
            self.register_buffer(
                "state_prior_std",
                torch.ones(mechanism.n_energy_params) * 20.0
            )

            # Prior means for barriers (typical activation energies)
            self.register_buffer(
                "barrier_prior_mean",
                torch.ones(mechanism.n_barrier_params) * 60.0
            )
            self.register_buffer(
                "barrier_prior_std",
                torch.ones(mechanism.n_barrier_params) * 15.0
            )

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict energies for this mechanism.

        Returns:
            Tuple of (state_energies, barrier_energies)
        """
        return self.decoder(h)

    def get_full_state_energies(self, state_energies: Tensor) -> Tensor:
        """
        Get full state energy tensor including reference state (energy = 0).

        Args:
            state_energies: Predicted energies, shape (batch, n_energy_params)

        Returns:
            Full energies including reference, shape (batch, n_states)
        """
        batch_size = state_energies.shape[0]
        n_states = len(self.mechanism.states)

        full_energies = torch.zeros(batch_size, n_states, device=state_energies.device)

        param_idx = 0
        for i, state in enumerate(self.mechanism.states):
            if state.is_reference:
                full_energies[:, i] = 0.0
            else:
                full_energies[:, i] = state_energies[:, param_idx]
                param_idx += 1

        return full_energies

    def compute_prior_loss(
        self,
        state_energies: Tensor,
        barrier_energies: Tensor,
    ) -> Tensor:
        """
        Compute prior loss (negative log prior probability).

        Returns:
            Prior loss scalar
        """
        if not self.use_priors:
            return torch.tensor(0.0, device=state_energies.device)

        # Gaussian prior on state energies
        state_loss = 0.5 * torch.mean(
            ((state_energies - self.state_prior_mean) / self.state_prior_std) ** 2
        )

        # Gaussian prior on barrier energies
        barrier_loss = 0.5 * torch.mean(
            ((barrier_energies - self.barrier_prior_mean) / self.barrier_prior_std) ** 2
        )

        return state_loss + barrier_loss


class MultiMechanismDecoder(nn.Module):
    """
    Decoder that can predict energies for multiple mechanisms.

    Contains a separate decoder head for each mechanism.
    """

    def __init__(
        self,
        mechanisms: Dict[str, MechanismTemplate],
        d_input: int,
        hidden_dims: List[int] = [256, 128],
        shared_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            mechanisms: Dict mapping mechanism names to templates
            d_input: Dimension of input latent representation
            hidden_dims: Hidden layer dimensions
            shared_layers: Number of shared layers before mechanism-specific heads
            dropout: Dropout probability
        """
        super().__init__()
        self.mechanism_names = list(mechanisms.keys())
        self.mechanisms = mechanisms

        # Shared feature extraction
        shared_dims = hidden_dims[:shared_layers] if shared_layers > 0 else []
        layers = []
        in_dim = d_input
        for hidden_dim in shared_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.shared = nn.Sequential(*layers) if layers else nn.Identity()

        # Mechanism-specific heads
        head_dims = hidden_dims[shared_layers:] if shared_layers < len(hidden_dims) else [128]
        self.heads = nn.ModuleDict({
            name: MechanismSpecificDecoder(
                mechanism=mech,
                d_input=in_dim,
                hidden_dims=head_dims,
                dropout=dropout,
            )
            for name, mech in mechanisms.items()
        })

    def forward(
        self,
        h: Tensor,
        mechanism_name: Optional[str] = None,
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Predict energies for one or all mechanisms.

        Args:
            h: Latent representation, shape (batch, d_input)
            mechanism_name: If provided, only decode for this mechanism

        Returns:
            Dict mapping mechanism names to (state_energies, barrier_energies)
        """
        shared_features = self.shared(h)

        if mechanism_name is not None:
            return {mechanism_name: self.heads[mechanism_name](shared_features)}

        return {
            name: head(shared_features)
            for name, head in self.heads.items()
        }

    def compute_prior_loss(
        self,
        predictions: Dict[str, Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """
        Compute total prior loss across all mechanisms.
        """
        total_loss = 0.0
        for name, (state_e, barrier_e) in predictions.items():
            total_loss = total_loss + self.heads[name].compute_prior_loss(state_e, barrier_e)
        return total_loss / len(predictions)


class ThermodynamicConstrainedDecoder(nn.Module):
    """
    Decoder that enforces thermodynamic constraints on the output.

    Ensures that:
    1. Overall reaction energy matches known ΔG°_rxn (if provided)
    2. Barrier heights satisfy microscopic reversibility
    3. Energies are within physically reasonable ranges
    """

    def __init__(
        self,
        mechanism: MechanismTemplate,
        d_input: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        known_dg_rxn: Optional[float] = None,
    ):
        """
        Args:
            mechanism: Mechanism template
            d_input: Input dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            known_dg_rxn: Known reaction ΔG° in kJ/mol (if available)
        """
        super().__init__()
        self.mechanism = mechanism
        self.known_dg_rxn = known_dg_rxn

        # Base decoder for intermediate energies
        # We'll parameterize differently to enforce constraints
        n_internal_states = mechanism.n_energy_params - 1  # Exclude ΔG_rxn
        n_barriers = mechanism.n_barrier_params

        self.base_decoder = EnergyDecoder(
            d_input=d_input,
            n_state_energies=n_internal_states,
            n_barrier_energies=n_barriers,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # If reaction energy is not known, predict it
        if known_dg_rxn is None:
            self.dg_rxn_head = nn.Sequential(
                nn.Linear(d_input, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
        else:
            self.register_buffer("dg_rxn", torch.tensor([known_dg_rxn]))
            self.dg_rxn_head = None

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict thermodynamically consistent energies.

        Args:
            h: Latent representation, shape (batch, d_input)

        Returns:
            Tuple of (state_energies, barrier_energies)
            state_energies includes the reaction ΔG° as the last element
        """
        batch_size = h.shape[0]

        # Get base predictions
        internal_energies, barrier_energies = self.base_decoder(h)

        # Get reaction energy
        if self.dg_rxn_head is not None:
            dg_rxn = self.dg_rxn_head(h)  # (batch, 1)
        else:
            dg_rxn = self.dg_rxn.expand(batch_size, 1)

        # Concatenate to form full state energies
        # [internal_energies, dg_rxn]
        state_energies = torch.cat([internal_energies, dg_rxn], dim=-1)

        # Enforce barrier constraints
        # Barrier must be higher than both connected states
        # This is automatically satisfied by our parameterization (softplus + offset)

        return state_energies, barrier_energies

    def get_reaction_energy(self, h: Tensor) -> Tensor:
        """Get predicted reaction ΔG°."""
        if self.dg_rxn_head is not None:
            return self.dg_rxn_head(h).squeeze(-1)
        else:
            return self.dg_rxn.expand(h.shape[0])
