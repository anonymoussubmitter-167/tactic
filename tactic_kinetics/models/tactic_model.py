"""
Main TACTIC-Kinetics model.

This module implements the complete thermodynamic-native inference model
that combines the encoder, decoders, classifier, and ODE simulator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from ..mechanisms.base import MechanismTemplate
from ..mechanisms.templates import get_mechanism_by_name, get_all_mechanisms
from .encoder import ObservationEncoder
from .decoder import MultiMechanismDecoder, MechanismSpecificDecoder
from .classifier import MechanismClassifier
from .ode_simulator import ODESimulator


class TACTICKinetics(nn.Module):
    """
    TACTIC-Kinetics: Thermodynamic-Native Inference for Enzyme Mechanism Discovery.

    This model combines:
    1. Transformer encoder for sparse observations
    2. Mechanism-specific energy decoders
    3. Mechanism classifier
    4. Differentiable ODE simulator

    The model can:
    - Infer mechanism type from progress curves
    - Predict Gibbs energy landscapes
    - Simulate kinetic trajectories
    - All while maintaining thermodynamic consistency
    """

    def __init__(
        self,
        mechanism_names: Optional[List[str]] = None,
        d_model: int = 256,
        n_encoder_layers: int = 6,
        n_encoder_heads: int = 8,
        d_ff: int = 1024,
        n_decoder_layers: List[int] = [256, 128],
        dropout: float = 0.1,
        n_conditions: int = 4,
        condition_names: Optional[List[str]] = None,
        temperature: float = 298.15,
        use_adjoint: bool = True,
    ):
        """
        Args:
            mechanism_names: List of mechanism names to support. If None, uses all.
            d_model: Dimension of the encoder model
            n_encoder_layers: Number of transformer encoder layers
            n_encoder_heads: Number of attention heads
            d_ff: Feed-forward dimension
            n_decoder_layers: Hidden dimensions for energy decoders
            dropout: Dropout probability
            n_conditions: Number of experimental conditions
            condition_names: Names of conditions (e.g., ["temperature", "pH", "S0", "E0"])
            temperature: Default temperature for simulations (K)
            use_adjoint: Whether to use adjoint method for ODE solver
        """
        super().__init__()

        # Get mechanisms
        if mechanism_names is None:
            self.mechanisms = get_all_mechanisms()
        else:
            self.mechanisms = {name: get_mechanism_by_name(name) for name in mechanism_names}

        self.mechanism_names = list(self.mechanisms.keys())
        self.n_mechanisms = len(self.mechanism_names)
        self.d_model = d_model
        self.temperature = temperature

        # Observation encoder
        self.encoder = ObservationEncoder(
            d_model=d_model,
            n_heads=n_encoder_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_observables=1,
            n_conditions=n_conditions,
            condition_names=condition_names,
        )

        # Multi-mechanism decoder
        self.decoder = MultiMechanismDecoder(
            mechanisms=self.mechanisms,
            d_input=d_model,
            hidden_dims=n_decoder_layers,
            dropout=dropout,
        )

        # Mechanism classifier
        self.classifier = MechanismClassifier(
            d_input=d_model,
            mechanism_names=self.mechanism_names,
            hidden_dims=[256, 128],
            dropout=dropout,
        )

        # ODE simulators for each mechanism
        self.simulators = nn.ModuleDict({
            name: ODESimulator(
                mechanism=mech,
                temperature=temperature,
                use_adjoint=use_adjoint,
            )
            for name, mech in self.mechanisms.items()
        })

    def encode(
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
            values: Observed values, shape (batch, n_obs)
            conditions: Experimental conditions, shape (batch, n_conditions)
            mask: Valid observation mask, shape (batch, n_obs)

        Returns:
            Latent representation, shape (batch, d_model)
        """
        return self.encoder(times, values, conditions, mask)

    def decode_energies(
        self,
        h: Tensor,
        mechanism_name: Optional[str] = None,
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Decode latent representation to energy parameters.

        Args:
            h: Latent representation, shape (batch, d_model)
            mechanism_name: If provided, only decode for this mechanism

        Returns:
            Dict mapping mechanism names to (state_energies, barrier_energies)
        """
        return self.decoder(h, mechanism_name)

    def classify_mechanism(self, h: Tensor) -> Tensor:
        """
        Classify mechanism from latent representation.

        Args:
            h: Latent representation, shape (batch, d_model)

        Returns:
            Mechanism logits, shape (batch, n_mechanisms)
        """
        return self.classifier(h)

    def simulate(
        self,
        state_energies: Tensor,
        barrier_energies: Tensor,
        mechanism_name: str,
        initial_conditions: Tensor,
        t_eval: Tensor,
    ) -> Tensor:
        """
        Simulate kinetics for a specific mechanism.

        Args:
            state_energies: shape (batch, n_states)
            barrier_energies: shape (batch, n_barriers)
            mechanism_name: Name of the mechanism
            initial_conditions: shape (batch, n_species)
            t_eval: Time points, shape (n_times,)

        Returns:
            Trajectory, shape (batch, n_times, n_species)
        """
        simulator = self.simulators[mechanism_name]
        return simulator(state_energies, barrier_energies, initial_conditions, t_eval)

    def forward(
        self,
        times: Tensor,
        values: Tensor,
        conditions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_energies: bool = True,
        return_mechanism_probs: bool = True,
    ) -> Dict[str, Union[Tensor, Dict]]:
        """
        Full forward pass.

        Args:
            times: Observation times, shape (batch, n_obs)
            values: Observed values, shape (batch, n_obs)
            conditions: Experimental conditions, shape (batch, n_conditions)
            mask: Valid observation mask, shape (batch, n_obs)
            return_energies: Whether to return energy predictions
            return_mechanism_probs: Whether to return mechanism probabilities

        Returns:
            Dict containing:
                "latent": Latent representation, shape (batch, d_model)
                "mechanism_logits": Mechanism logits (if return_mechanism_probs)
                "mechanism_probs": Mechanism probabilities (if return_mechanism_probs)
                "energies": Dict of energy predictions (if return_energies)
        """
        # Encode observations
        h = self.encode(times, values, conditions, mask)

        result = {"latent": h}

        # Classify mechanism
        if return_mechanism_probs:
            logits = self.classify_mechanism(h)
            result["mechanism_logits"] = logits
            result["mechanism_probs"] = F.softmax(logits, dim=-1)

        # Decode energies
        if return_energies:
            result["energies"] = self.decode_energies(h)

        return result

    def infer_and_simulate(
        self,
        times: Tensor,
        values: Tensor,
        conditions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        t_eval: Optional[Tensor] = None,
        initial_conditions: Optional[Dict[str, Tensor]] = None,
        mechanism_name: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """
        Full inference and simulation pipeline.

        If mechanism_name is not provided, uses the most likely mechanism.

        Args:
            times: Observation times, shape (batch, n_obs)
            values: Observed values, shape (batch, n_obs)
            conditions: Experimental conditions, shape (batch, n_conditions)
            mask: Valid observation mask
            t_eval: Time points for simulation. If None, uses observation times.
            initial_conditions: Dict mapping mechanism names to initial conditions
            mechanism_name: Override mechanism selection

        Returns:
            Dict containing:
                "predicted_mechanism": Predicted mechanism name
                "mechanism_confidence": Confidence score
                "trajectory": Simulated trajectory
                "state_energies": Predicted state energies
                "barrier_energies": Predicted barrier energies
        """
        batch_size = times.shape[0]

        # Forward pass
        outputs = self.forward(times, values, conditions, mask)

        # Get mechanism
        if mechanism_name is None:
            probs = outputs["mechanism_probs"]
            confidence, idx = probs.max(dim=-1)
            mechanism_name = self.mechanism_names[idx[0].item()]  # Assume same for batch
        else:
            confidence = torch.ones(batch_size, device=times.device)

        # Get energies for selected mechanism
        state_e, barrier_e = outputs["energies"][mechanism_name]

        # Get full state energies (including reference)
        decoder = self.decoder.heads[mechanism_name]
        full_state_e = decoder.get_full_state_energies(state_e)

        # Simulation time points
        if t_eval is None:
            t_eval = times[0]  # Use first sample's times

        # Initial conditions
        if initial_conditions is None:
            # Default initial conditions
            simulator = self.simulators[mechanism_name]
            n_species = simulator.n_species
            ic = torch.zeros(batch_size, n_species, device=times.device)
            # Set substrate concentration from conditions if available
            if conditions is not None and conditions.shape[1] >= 3:
                ic[:, simulator.ode.species_idx.get("S", 0)] = conditions[:, 2]  # Assume S0 is 3rd condition
            else:
                ic[:, 0] = 1.0  # Default substrate
        else:
            ic = initial_conditions.get(mechanism_name)

        # Simulate
        trajectory = self.simulate(full_state_e, barrier_e, mechanism_name, ic, t_eval)

        return {
            "predicted_mechanism": mechanism_name,
            "mechanism_confidence": confidence,
            "trajectory": trajectory,
            "state_energies": state_e,
            "barrier_energies": barrier_e,
            "full_state_energies": full_state_e,
        }


class TACTICLoss(nn.Module):
    """
    Combined loss function for TACTIC-Kinetics training.

    Combines:
    1. Trajectory reconstruction loss
    2. Mechanism classification loss
    3. Thermodynamic consistency loss
    4. Energy prior regularization
    """

    def __init__(
        self,
        lambda_traj: float = 1.0,
        lambda_mech: float = 1.0,
        lambda_thermo: float = 0.1,
        lambda_prior: float = 0.01,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_traj = lambda_traj
        self.lambda_mech = lambda_mech
        self.lambda_thermo = lambda_thermo
        self.lambda_prior = lambda_prior
        self.label_smoothing = label_smoothing

    def trajectory_loss(
        self,
        predicted: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute trajectory reconstruction loss.

        Args:
            predicted: Predicted trajectory, shape (batch, n_times, ...)
            target: Target trajectory, shape (batch, n_times, ...)
            mask: Valid time points mask, shape (batch, n_times)

        Returns:
            Loss scalar
        """
        mse = (predicted - target) ** 2

        if mask is not None:
            # Expand mask to match dimensions
            while mask.dim() < mse.dim():
                mask = mask.unsqueeze(-1)
            mse = mse * mask
            n_valid = mask.sum()
            loss = mse.sum() / (n_valid + 1e-8)
        else:
            loss = mse.mean()

        return loss

    def mechanism_loss(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute mechanism classification loss.

        Args:
            logits: Mechanism logits, shape (batch, n_mechanisms)
            labels: Ground truth labels, shape (batch,)

        Returns:
            Loss scalar
        """
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

    def thermodynamic_loss(
        self,
        predicted_dg: Tensor,
        known_dg: Tensor,
        uncertainty: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute thermodynamic consistency loss.

        Args:
            predicted_dg: Predicted reaction ΔG°, shape (batch,)
            known_dg: Known ΔG° from eQuilibrator, shape (batch,)
            uncertainty: Uncertainty in known values, shape (batch,)

        Returns:
            Loss scalar
        """
        diff = predicted_dg - known_dg

        if uncertainty is not None:
            # Weighted by inverse uncertainty
            loss = (diff ** 2 / (uncertainty ** 2 + 1e-8)).mean()
        else:
            loss = (diff ** 2).mean()

        return loss

    def prior_loss(
        self,
        state_energies: Tensor,
        barrier_energies: Tensor,
        state_prior_mean: Optional[Tensor] = None,
        state_prior_std: Optional[Tensor] = None,
        barrier_prior_mean: Optional[Tensor] = None,
        barrier_prior_std: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute energy prior regularization loss.

        Args:
            state_energies: Predicted state energies
            barrier_energies: Predicted barrier energies
            *_prior_*: Prior statistics (if None, uses defaults)

        Returns:
            Loss scalar
        """
        # Default priors
        if state_prior_mean is None:
            state_prior_mean = torch.zeros_like(state_energies)
        if state_prior_std is None:
            state_prior_std = torch.ones_like(state_energies) * 20.0

        if barrier_prior_mean is None:
            barrier_prior_mean = torch.ones_like(barrier_energies) * 60.0
        if barrier_prior_std is None:
            barrier_prior_std = torch.ones_like(barrier_energies) * 15.0

        state_loss = 0.5 * ((state_energies - state_prior_mean) / state_prior_std) ** 2
        barrier_loss = 0.5 * ((barrier_energies - barrier_prior_mean) / barrier_prior_std) ** 2

        return state_loss.mean() + barrier_loss.mean()

    def forward(
        self,
        predictions: Dict,
        targets: Dict,
    ) -> Dict[str, Tensor]:
        """
        Compute total loss and components.

        Args:
            predictions: Dict from model forward pass
            targets: Dict containing:
                "trajectory": Target trajectory (optional)
                "mechanism_labels": Mechanism labels (optional)
                "known_dg": Known ΔG° values (optional)

        Returns:
            Dict with "total" and individual loss components
        """
        losses = {}
        total = 0.0

        # Trajectory loss
        if "trajectory" in predictions and "trajectory" in targets:
            traj_loss = self.trajectory_loss(
                predictions["trajectory"],
                targets["trajectory"],
                targets.get("trajectory_mask"),
            )
            losses["trajectory"] = traj_loss
            total = total + self.lambda_traj * traj_loss

        # Mechanism loss
        if "mechanism_logits" in predictions and "mechanism_labels" in targets:
            mech_loss = self.mechanism_loss(
                predictions["mechanism_logits"],
                targets["mechanism_labels"],
            )
            losses["mechanism"] = mech_loss
            total = total + self.lambda_mech * mech_loss

        # Thermodynamic loss
        if "predicted_dg" in predictions and "known_dg" in targets:
            thermo_loss = self.thermodynamic_loss(
                predictions["predicted_dg"],
                targets["known_dg"],
                targets.get("dg_uncertainty"),
            )
            losses["thermodynamic"] = thermo_loss
            total = total + self.lambda_thermo * thermo_loss

        # Prior loss
        if "state_energies" in predictions and "barrier_energies" in predictions:
            prior_loss = self.prior_loss(
                predictions["state_energies"],
                predictions["barrier_energies"],
            )
            losses["prior"] = prior_loss
            total = total + self.lambda_prior * prior_loss

        losses["total"] = total
        return losses
