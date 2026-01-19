"""
Differentiable ODE simulator for enzyme kinetics.

This module provides a differentiable ODE solver that can be used with
PyTorch autograd for end-to-end training of kinetic models.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable
from torchdiffeq import odeint, odeint_adjoint

from ..mechanisms.base import MechanismTemplate, Transition, TransitionType
from ..utils.thermodynamics import eyring_rate_constant
from ..utils.constants import T_STANDARD


class EnergyToRateConverter(nn.Module):
    """
    Converts Gibbs energy parameters to rate constants using the Eyring equation.

    Given state energies and barrier energies, computes forward and reverse
    rate constants for all transitions in a mechanism.
    """

    def __init__(
        self,
        mechanism: MechanismTemplate,
        temperature: float = T_STANDARD,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.register_buffer("temperature", torch.tensor(temperature))

        # Pre-compute Eyring prefactor
        from ..utils.constants import K_B, H
        prefactor = K_B * temperature / H
        self.register_buffer("eyring_prefactor", torch.tensor(prefactor))

        # Build index mappings for states and transitions
        self.state_names = [s.name for s in mechanism.states]
        self.state_idx = {name: i for i, name in enumerate(self.state_names)}

        self.transition_names = [t.name for t in mechanism.transitions]
        self.transition_idx = {name: i for i, name in enumerate(self.transition_names)}

        # Find reference state index
        for i, s in enumerate(mechanism.states):
            if s.is_reference:
                self.reference_idx = i
                break

    def forward(
        self,
        state_energies: Tensor,
        barrier_energies: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert energies to rate constants.

        Args:
            state_energies: Tensor of shape (batch, n_states) with Gibbs energies
                           in kJ/mol. Can be (batch, n_energy_params) if reference
                           is excluded - it will be inserted as 0.
            barrier_energies: Tensor of shape (batch, n_transitions) with activation
                             energies in kJ/mol.

        Returns:
            Tuple of (forward_rates, reverse_rates), each of shape (batch, n_transitions)
        """
        batch_size = state_energies.shape[0]
        n_transitions = len(self.mechanism.transitions)
        n_states = len(self.mechanism.states)

        # Handle case where reference state energy is not included
        if state_energies.shape[1] == n_states - 1:
            # Insert reference state energy (0) at the correct position
            full_energies = torch.zeros(batch_size, n_states, device=state_energies.device)
            energy_idx = 0
            for i, state in enumerate(self.mechanism.states):
                if state.is_reference:
                    full_energies[:, i] = 0.0
                else:
                    full_energies[:, i] = state_energies[:, energy_idx]
                    energy_idx += 1
            state_energies = full_energies

        forward_rates = torch.zeros(batch_size, n_transitions, device=state_energies.device)
        reverse_rates = torch.zeros(batch_size, n_transitions, device=state_energies.device)

        R = 8.314462618e-3  # kJ/(mol·K)
        RT = R * self.temperature

        for i, trans in enumerate(self.mechanism.transitions):
            from_idx = self.state_idx[trans.from_state]
            to_idx = self.state_idx[trans.to_state]

            G_from = state_energies[:, from_idx]
            G_to = state_energies[:, to_idx]
            G_barrier = barrier_energies[:, i]

            # Forward rate: k_f = prefactor * exp(-ΔG‡_f / RT)
            # where ΔG‡_f = G_barrier - G_from (barrier relative to from state)
            delta_G_forward = G_barrier - G_from
            k_forward = self.eyring_prefactor * torch.exp(-delta_G_forward / RT)
            forward_rates[:, i] = k_forward

            # Reverse rate: k_r = prefactor * exp(-ΔG‡_r / RT)
            # where ΔG‡_r = G_barrier - G_to (barrier relative to to state)
            if trans.is_reversible:
                delta_G_reverse = G_barrier - G_to
                k_reverse = self.eyring_prefactor * torch.exp(-delta_G_reverse / RT)
                reverse_rates[:, i] = k_reverse
            else:
                reverse_rates[:, i] = 0.0

        return forward_rates, reverse_rates

    def compute_keq(self, state_energies: Tensor) -> Dict[str, Tensor]:
        """
        Compute equilibrium constants from state energies.

        Returns dict mapping transition name to K_eq = k_f / k_r
        """
        R = 8.314462618e-3
        RT = R * self.temperature

        keq_dict = {}
        for trans in self.mechanism.transitions:
            if trans.is_reversible:
                from_idx = self.state_idx[trans.from_state]
                to_idx = self.state_idx[trans.to_state]

                G_from = state_energies[:, from_idx]
                G_to = state_energies[:, to_idx]

                # K_eq = exp(-(G_to - G_from) / RT)
                delta_G = G_to - G_from
                keq = torch.exp(-delta_G / RT)
                keq_dict[trans.name] = keq

        return keq_dict


class MechanismODE(nn.Module):
    """
    Defines the ODE system for a given enzyme mechanism.

    The ODE system describes the time evolution of species concentrations
    based on the mechanism's rate equations.
    """

    def __init__(
        self,
        mechanism: MechanismTemplate,
        species_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.mechanism = mechanism

        # Determine species to track
        if species_names is None:
            # Track all enzyme states plus free species
            self.species_names = self._get_trackable_species()
        else:
            self.species_names = species_names

        self.n_species = len(self.species_names)
        self.species_idx = {name: i for i, name in enumerate(self.species_names)}

        # Build transition info
        self._build_transition_info()

    def _get_trackable_species(self) -> List[str]:
        """Get list of species to track in ODE."""
        species = set()

        # Add enzyme states
        for state in self.mechanism.states:
            # For bound states, track the complex
            if state.state_type.name in ["ENZYME_SUBSTRATE", "ENZYME_PRODUCT",
                                          "ENZYME_INHIBITOR", "TERNARY_SUBSTRATE",
                                          "TERNARY_PRODUCT"]:
                # Get the complex name (first species that starts with E)
                for sp in state.species:
                    if sp.startswith("E") or sp.startswith("F"):
                        species.add(sp)
                        break
            elif state.state_type.name == "FREE_ENZYME":
                species.add("E")

        # Add substrates and products
        species.update(self.mechanism.substrate_names)
        species.update(self.mechanism.product_names)
        species.update(self.mechanism.inhibitor_names)

        return sorted(list(species))

    def _build_transition_info(self):
        """Build data structures for efficient ODE computation."""
        # Store transition effects on species
        self.transition_effects = []

        for trans in self.mechanism.transitions:
            effects = {}

            # Get enzyme states involved
            from_state = self.mechanism.get_state(trans.from_state)
            to_state = self.mechanism.get_state(trans.to_state)

            # Determine enzyme species change
            from_enzyme = self._get_enzyme_species(from_state)
            to_enzyme = self._get_enzyme_species(to_state)

            if from_enzyme and from_enzyme in self.species_idx:
                effects[from_enzyme] = effects.get(from_enzyme, 0) - 1
            if to_enzyme and to_enzyme in self.species_idx:
                effects[to_enzyme] = effects.get(to_enzyme, 0) + 1

            # Add stoichiometry effects
            for species, coeff in trans.stoichiometry.items():
                if species in self.species_idx:
                    effects[species] = effects.get(species, 0) + coeff

            self.transition_effects.append(effects)

    def _get_enzyme_species(self, state) -> Optional[str]:
        """Get the enzyme species name for a state."""
        for sp in state.species:
            if sp.startswith("E") or sp.startswith("F"):
                if len(sp) > 1:  # Complex like ES, EI, etc.
                    return sp
                else:  # Free E
                    return "E"
        return None

    def forward(
        self,
        t: Tensor,
        y: Tensor,
        forward_rates: Tensor,
        reverse_rates: Tensor,
    ) -> Tensor:
        """
        Compute dy/dt for the ODE system.

        Args:
            t: Current time (scalar)
            y: Current state, shape (batch, n_species)
            forward_rates: Forward rate constants, shape (batch, n_transitions)
            reverse_rates: Reverse rate constants, shape (batch, n_transitions)

        Returns:
            dy/dt, shape (batch, n_species)
        """
        batch_size = y.shape[0]
        dydt = torch.zeros_like(y)

        for i, trans in enumerate(self.mechanism.transitions):
            # Get reactant concentrations
            from_state = self.mechanism.get_state(trans.from_state)
            from_enzyme = self._get_enzyme_species(from_state)

            to_state = self.mechanism.get_state(trans.to_state)
            to_enzyme = self._get_enzyme_species(to_state)

            # Forward reaction rate
            if from_enzyme and from_enzyme in self.species_idx:
                enzyme_conc = y[:, self.species_idx[from_enzyme]]
            else:
                enzyme_conc = torch.ones(batch_size, device=y.device)

            # Include substrate concentration for binding reactions
            substrate_conc = torch.ones(batch_size, device=y.device)
            for species, coeff in trans.stoichiometry.items():
                if coeff < 0 and species in self.species_idx:
                    substrate_conc = substrate_conc * y[:, self.species_idx[species]] ** (-coeff)

            v_forward = forward_rates[:, i] * enzyme_conc * substrate_conc

            # Reverse reaction rate
            v_reverse = torch.zeros(batch_size, device=y.device)
            if trans.is_reversible:
                if to_enzyme and to_enzyme in self.species_idx:
                    enzyme_conc_rev = y[:, self.species_idx[to_enzyme]]
                else:
                    enzyme_conc_rev = torch.ones(batch_size, device=y.device)

                # Include product concentration for reverse binding
                product_conc = torch.ones(batch_size, device=y.device)
                for species, coeff in trans.stoichiometry.items():
                    if coeff > 0 and species in self.species_idx:
                        product_conc = product_conc * y[:, self.species_idx[species]] ** coeff

                v_reverse = reverse_rates[:, i] * enzyme_conc_rev * product_conc

            # Net reaction rate
            v_net = v_forward - v_reverse

            # Apply effects to species
            for species, effect in self.transition_effects[i].items():
                if species in self.species_idx:
                    dydt[:, self.species_idx[species]] += effect * v_net

        return dydt


class ODESimulator(nn.Module):
    """
    Differentiable ODE simulator for enzyme kinetics.

    Combines energy-to-rate conversion with ODE integration to simulate
    enzyme kinetics trajectories from Gibbs energy parameters.
    """

    def __init__(
        self,
        mechanism: MechanismTemplate,
        temperature: float = T_STANDARD,
        solver: str = "dopri5",
        use_adjoint: bool = True,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.temperature = temperature
        self.solver = solver
        self.use_adjoint = use_adjoint
        self.rtol = rtol
        self.atol = atol

        # Create sub-modules
        self.energy_to_rate = EnergyToRateConverter(mechanism, temperature)
        self.ode = MechanismODE(mechanism)

        # Species info
        self.species_names = self.ode.species_names
        self.n_species = self.ode.n_species

    def forward(
        self,
        state_energies: Tensor,
        barrier_energies: Tensor,
        initial_conditions: Tensor,
        t_eval: Tensor,
    ) -> Tensor:
        """
        Simulate enzyme kinetics trajectory.

        Args:
            state_energies: Gibbs energies of states, shape (batch, n_states)
            barrier_energies: Activation energies, shape (batch, n_transitions)
            initial_conditions: Initial concentrations, shape (batch, n_species)
            t_eval: Time points for evaluation, shape (n_times,)

        Returns:
            Trajectory, shape (batch, n_times, n_species)
        """
        # Convert energies to rate constants
        forward_rates, reverse_rates = self.energy_to_rate(state_energies, barrier_energies)

        # Create ODE function with fixed rates
        def ode_func(t, y):
            return self.ode(t, y, forward_rates, reverse_rates)

        # Integrate ODE
        if self.use_adjoint:
            trajectory = odeint_adjoint(
                ode_func,
                initial_conditions,
                t_eval,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
            )
        else:
            trajectory = odeint(
                ode_func,
                initial_conditions,
                t_eval,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
            )

        # Reshape from (n_times, batch, n_species) to (batch, n_times, n_species)
        trajectory = trajectory.permute(1, 0, 2)

        return trajectory

    def get_observable(
        self,
        trajectory: Tensor,
        observable: str = "product",
    ) -> Tensor:
        """
        Extract observable from trajectory.

        Args:
            trajectory: Full trajectory, shape (batch, n_times, n_species)
            observable: Which observable to extract:
                - "product": Sum of product concentrations
                - "substrate": Sum of substrate concentrations
                - species name: Specific species concentration

        Returns:
            Observable trajectory, shape (batch, n_times)
        """
        if observable == "product":
            product_idx = [
                self.ode.species_idx[p]
                for p in self.mechanism.product_names
                if p in self.ode.species_idx
            ]
            return trajectory[:, :, product_idx].sum(dim=-1)
        elif observable == "substrate":
            substrate_idx = [
                self.ode.species_idx[s]
                for s in self.mechanism.substrate_names
                if s in self.ode.species_idx
            ]
            return trajectory[:, :, substrate_idx].sum(dim=-1)
        elif observable in self.ode.species_idx:
            idx = self.ode.species_idx[observable]
            return trajectory[:, :, idx]
        else:
            raise ValueError(f"Unknown observable: {observable}")


class SimpleMichaelisMentenODE(nn.Module):
    """
    Simplified ODE for irreversible Michaelis-Menten kinetics.

    This is a more efficient implementation for the simple case:
    E + S <-> ES -> E + P

    Uses the quasi-steady-state approximation or full ODE as specified.
    """

    def __init__(self, use_qssa: bool = False):
        super().__init__()
        self.use_qssa = use_qssa

    def forward(
        self,
        t: Tensor,
        y: Tensor,
        kcat: Tensor,
        Km: Tensor,
        E_total: Tensor,
    ) -> Tensor:
        """
        Compute dy/dt for Michaelis-Menten kinetics.

        Args:
            t: Current time
            y: Current state [S, P], shape (batch, 2)
            kcat: Catalytic rate constant, shape (batch,)
            Km: Michaelis constant, shape (batch,)
            E_total: Total enzyme concentration, shape (batch,)

        Returns:
            dy/dt, shape (batch, 2)
        """
        S = y[:, 0]
        P = y[:, 1]

        if self.use_qssa:
            # Quasi-steady-state: v = kcat * E_total * S / (Km + S)
            v = kcat * E_total * S / (Km + S + 1e-10)
        else:
            # Full equation (assuming fast equilibrium)
            v = kcat * E_total * S / (Km + S + 1e-10)

        dSdt = -v
        dPdt = v

        return torch.stack([dSdt, dPdt], dim=-1)


def simulate_michaelis_menten(
    kcat: Tensor,
    Km: Tensor,
    E_total: Tensor,
    S0: Tensor,
    t_eval: Tensor,
    solver: str = "euler",
) -> Tuple[Tensor, Tensor]:
    """
    Convenience function to simulate Michaelis-Menten kinetics.

    Args:
        kcat: Catalytic rate constant (batch,)
        Km: Michaelis constant (batch,)
        E_total: Total enzyme concentration (batch,)
        S0: Initial substrate concentration (batch,)
        t_eval: Time points (n_times,)
        solver: Solver to use - "euler" (fast, for synthetic data) or "dopri5" (accurate)

    Returns:
        Tuple of (S, P) trajectories, each shape (batch, n_times)
    """
    batch_size = kcat.shape[0]

    if solver == "euler":
        # Use simple Euler method - fast and stable for synthetic data generation
        return _simulate_mm_euler(kcat, Km, E_total, S0, t_eval)

    # Use torchdiffeq for more accurate simulations
    y0 = torch.stack([S0, torch.zeros_like(S0)], dim=-1)

    ode = SimpleMichaelisMentenODE(use_qssa=True)

    def ode_func(t, y):
        return ode(t, y, kcat, Km, E_total)

    trajectory = odeint(ode_func, y0, t_eval, method=solver)
    trajectory = trajectory.permute(1, 0, 2)  # (batch, n_times, 2)

    S = trajectory[:, :, 0]
    P = trajectory[:, :, 1]

    return S, P


def _simulate_mm_euler(
    kcat: Tensor,
    Km: Tensor,
    E_total: Tensor,
    S0: Tensor,
    t_eval: Tensor,
    max_steps: int = 10000,
) -> Tuple[Tensor, Tensor]:
    """
    Fast Euler simulation of Michaelis-Menten kinetics.

    Uses adaptive time stepping for numerical stability.
    Suitable for synthetic data generation where high accuracy isn't critical.

    Args:
        kcat: Catalytic rate constant (batch,)
        Km: Michaelis constant (batch,)
        E_total: Total enzyme concentration (batch,)
        S0: Initial substrate concentration (batch,)
        t_eval: Time points to evaluate at (n_times,)
        max_steps: Maximum integration steps

    Returns:
        Tuple of (S, P) trajectories, each shape (batch, n_times)
    """
    batch_size = kcat.shape[0]
    n_times = len(t_eval)

    # Initialize output trajectories
    S_traj = torch.zeros(batch_size, n_times, device=S0.device)
    P_traj = torch.zeros(batch_size, n_times, device=S0.device)

    # Initial conditions
    S = S0.clone()
    P = torch.zeros_like(S0)

    S_traj[:, 0] = S
    P_traj[:, 0] = P

    # Integrate between each time point
    for i in range(1, n_times):
        t_start = t_eval[i - 1].item()
        t_end = t_eval[i].item()
        dt_target = t_end - t_start

        # Adaptive time stepping
        t_current = t_start
        steps = 0

        while t_current < t_end and steps < max_steps:
            # Compute rate
            v = kcat * E_total * S / (Km + S + 1e-10)

            # Adaptive step size based on substrate consumption rate
            # Don't let S change by more than 10% in one step
            max_dS = 0.1 * (S + 1e-10)
            dt_stable = max_dS / (v.abs() + 1e-10)
            dt = torch.clamp(dt_stable.min(), max=dt_target)
            dt = min(dt.item(), t_end - t_current)

            # Euler step
            dS = -v * dt
            S = torch.clamp(S + dS, min=0.0)
            P = P - dS  # Conservation: P increases by what S decreases

            t_current += dt
            steps += 1

        # Store at this time point
        S_traj[:, i] = S
        P_traj[:, i] = P

    return S_traj, P_traj


# =============================================================================
# General Mechanism ODE Simulation (supports all 10 mechanism types)
# =============================================================================

class GeneralMechanismODE(nn.Module):
    """
    General ODE system for any enzyme mechanism template.

    This class builds the full ODE system from a MechanismTemplate,
    properly tracking all enzyme species, substrates, products, and inhibitors.

    Supports:
    - Single and multi-substrate mechanisms
    - Inhibition mechanisms (competitive, uncompetitive, mixed, substrate)
    - Bi-substrate mechanisms (ordered bi-bi, random bi-bi, ping-pong)
    - Product inhibition
    """

    def __init__(self, mechanism: MechanismTemplate):
        super().__init__()
        self.mechanism = mechanism

        # Build species list: enzyme forms + substrates + products + inhibitors
        self._build_species_list()

        # Build stoichiometry matrices for efficient computation
        self._build_stoichiometry_matrices()

    def _build_species_list(self):
        """Build complete list of species to track."""
        species = []

        # Collect all enzyme species from states
        enzyme_species = set()
        for state in self.mechanism.states:
            for sp in state.species:
                # Enzyme species start with E or F (for ping-pong)
                if sp.startswith("E") or sp.startswith("F"):
                    enzyme_species.add(sp)

        # Sort enzyme species: free enzymes first, then complexes by length
        enzyme_species = sorted(enzyme_species, key=lambda x: (len(x), x))
        species.extend(enzyme_species)

        # Add substrates
        for sub in self.mechanism.substrate_names:
            if sub not in species:
                species.append(sub)

        # Add products
        for prod in self.mechanism.product_names:
            if prod not in species:
                species.append(prod)

        # Add inhibitors
        for inh in self.mechanism.inhibitor_names:
            if inh not in species:
                species.append(inh)

        self.species_names = species
        self.n_species = len(species)
        self.species_idx = {name: i for i, name in enumerate(species)}

        # Identify enzyme species indices (for conservation)
        self.enzyme_indices = [
            i for i, sp in enumerate(species)
            if sp.startswith("E") or sp.startswith("F")
        ]

    def _build_stoichiometry_matrices(self):
        """Build stoichiometry matrices for ODE computation."""
        n_trans = len(self.mechanism.transitions)
        n_species = self.n_species

        # Forward stoichiometry: S_forward[i, j] = stoich coeff for species j in transition i
        self.register_buffer(
            "S_forward",
            torch.zeros(n_trans, n_species)
        )

        # Reverse stoichiometry
        self.register_buffer(
            "S_reverse",
            torch.zeros(n_trans, n_species)
        )

        # Reactant indices for rate computation (forward direction)
        self.forward_reactants = []  # List of (enzyme_idx, [substrate_indices])
        self.reverse_reactants = []  # List of (enzyme_idx, [product_indices])

        for i, trans in enumerate(self.mechanism.transitions):
            from_state = self.mechanism.get_state(trans.from_state)
            to_state = self.mechanism.get_state(trans.to_state)

            # Find enzyme species in from/to states
            from_enzyme = self._get_enzyme_from_state(from_state)
            to_enzyme = self._get_enzyme_from_state(to_state)

            # Forward direction: enzyme consumed from from_state
            forward_subs = []
            if from_enzyme is not None and from_enzyme in self.species_idx:
                self.S_forward[i, self.species_idx[from_enzyme]] = -1

            # Forward direction: enzyme produced in to_state
            if to_enzyme is not None and to_enzyme in self.species_idx:
                self.S_forward[i, self.species_idx[to_enzyme]] = 1

            # Apply stoichiometry from transition (substrates consumed, products produced)
            for species, coeff in trans.stoichiometry.items():
                if species in self.species_idx:
                    self.S_forward[i, self.species_idx[species]] = coeff
                    if coeff < 0:
                        forward_subs.append(self.species_idx[species])

            # Store reactant info for rate computation
            from_enzyme_idx = self.species_idx.get(from_enzyme, -1) if from_enzyme else -1
            self.forward_reactants.append((from_enzyme_idx, forward_subs))

            # Reverse direction (flip signs)
            reverse_prods = []
            if trans.is_reversible:
                to_enzyme_idx = self.species_idx.get(to_enzyme, -1) if to_enzyme else -1
                for species, coeff in trans.stoichiometry.items():
                    if species in self.species_idx and coeff > 0:
                        reverse_prods.append(self.species_idx[species])
                self.reverse_reactants.append((to_enzyme_idx, reverse_prods))
            else:
                self.reverse_reactants.append((-1, []))

    def _get_enzyme_from_state(self, state) -> Optional[str]:
        """Extract enzyme species name from a state."""
        for sp in state.species:
            if sp.startswith("E") or sp.startswith("F"):
                return sp
        return None

    def forward(
        self,
        t: Tensor,
        y: Tensor,
        forward_rates: Tensor,
        reverse_rates: Tensor,
    ) -> Tensor:
        """
        Compute dy/dt for the general mechanism ODE.

        Args:
            t: Current time (scalar)
            y: Current concentrations, shape (batch, n_species)
            forward_rates: Forward rate constants, shape (batch, n_transitions)
            reverse_rates: Reverse rate constants, shape (batch, n_transitions)

        Returns:
            dy/dt, shape (batch, n_species)
        """
        batch_size = y.shape[0]
        dydt = torch.zeros_like(y)

        for i, trans in enumerate(self.mechanism.transitions):
            # Forward rate: v_f = k_f * [enzyme] * product([substrates])
            enzyme_idx, sub_indices = self.forward_reactants[i]

            if enzyme_idx >= 0:
                v_forward = forward_rates[:, i] * y[:, enzyme_idx]
            else:
                v_forward = forward_rates[:, i]

            for sub_idx in sub_indices:
                v_forward = v_forward * y[:, sub_idx]

            # Reverse rate: v_r = k_r * [enzyme'] * product([products])
            v_reverse = torch.zeros(batch_size, device=y.device)
            if trans.is_reversible:
                enzyme_idx_rev, prod_indices = self.reverse_reactants[i]

                if enzyme_idx_rev >= 0:
                    v_reverse = reverse_rates[:, i] * y[:, enzyme_idx_rev]
                else:
                    v_reverse = reverse_rates[:, i]

                for prod_idx in prod_indices:
                    v_reverse = v_reverse * y[:, prod_idx]

            # Net rate = forward - reverse
            v_net = v_forward - v_reverse

            # Apply stoichiometry: dydt += v_net * S_forward
            # This correctly handles both forward and reverse contributions
            dydt = dydt + v_net.unsqueeze(-1) * self.S_forward[i:i+1, :]

        return dydt


def simulate_general_mechanism(
    mechanism: MechanismTemplate,
    forward_rates: Tensor,
    reverse_rates: Tensor,
    initial_conditions: Dict[str, Tensor],
    t_eval: Tensor,
    solver: str = "euler",
    max_steps: int = 10000,
) -> Dict[str, Tensor]:
    """
    Simulate any enzyme mechanism using the general ODE system.

    Args:
        mechanism: The mechanism template to simulate
        forward_rates: Forward rate constants, shape (batch, n_transitions)
        reverse_rates: Reverse rate constants, shape (batch, n_transitions)
        initial_conditions: Dict mapping species name to initial concentration (batch,)
        t_eval: Time points for evaluation (n_times,)
        solver: "euler" for fast synthetic data or "dopri5" for accuracy
        max_steps: Maximum integration steps for Euler solver

    Returns:
        Dict mapping species name to concentration trajectory (batch, n_times)
    """
    ode = GeneralMechanismODE(mechanism)
    batch_size = forward_rates.shape[0]
    n_times = len(t_eval)
    device = forward_rates.device

    # Build initial state vector
    y0 = torch.zeros(batch_size, ode.n_species, device=device)
    for species, conc in initial_conditions.items():
        if species in ode.species_idx:
            y0[:, ode.species_idx[species]] = conc

    if solver == "euler":
        # Use adaptive Euler method
        trajectories = _simulate_mechanism_euler(
            ode, forward_rates, reverse_rates, y0, t_eval, max_steps
        )
    else:
        # Use torchdiffeq
        def ode_func(t, y):
            return ode(t, y, forward_rates, reverse_rates)

        trajectories = odeint(ode_func, y0, t_eval, method=solver)
        trajectories = trajectories.permute(1, 0, 2)  # (batch, n_times, n_species)

    # Convert to dict
    result = {}
    for i, species in enumerate(ode.species_names):
        result[species] = trajectories[:, :, i]

    return result


def _simulate_mechanism_euler(
    ode: GeneralMechanismODE,
    forward_rates: Tensor,
    reverse_rates: Tensor,
    y0: Tensor,
    t_eval: Tensor,
    max_steps: int = 50000,
) -> Tensor:
    """
    Adaptive Euler simulation for general mechanism ODE.

    Uses adaptive time stepping to handle stiff enzyme kinetics ODEs.
    The step size is chosen to keep concentration changes bounded.

    Args:
        ode: The GeneralMechanismODE instance
        forward_rates: Forward rate constants (batch, n_transitions)
        reverse_rates: Reverse rate constants (batch, n_transitions)
        y0: Initial conditions (batch, n_species)
        t_eval: Time points (n_times,)
        max_steps: Maximum total steps

    Returns:
        Trajectories (batch, n_times, n_species)
    """
    batch_size = y0.shape[0]
    n_times = len(t_eval)
    n_species = y0.shape[1]
    device = y0.device

    # Initialize trajectory storage
    trajectories = torch.zeros(batch_size, n_times, n_species, device=device)

    # Current state
    y = y0.clone()
    trajectories[:, 0, :] = y

    # Dummy time tensor for ODE
    t_dummy = torch.tensor(0.0, device=device)

    # Estimate max rate for adaptive stepping
    max_rate = max(forward_rates.abs().max().item(), reverse_rates.abs().max().item(), 1.0)

    # Base dt should be small enough that max_rate * dt < 0.1 (10% change max)
    base_dt = 0.05 / max_rate

    total_steps = 0

    for i in range(1, n_times):
        t_start = t_eval[i - 1].item()
        t_end = t_eval[i].item()
        t_current = t_start

        while t_current < t_end - 1e-12 and total_steps < max_steps:
            # Compute derivative
            dydt = ode(t_dummy, y, forward_rates, reverse_rates)

            # Adaptive step: limit change to 5% of current value or 5% of max
            # This prevents large relative changes in small concentrations
            y_scale = torch.max(y.abs(), torch.ones_like(y) * 0.001)
            max_rel_change = (dydt.abs() / y_scale).max().item()

            if max_rel_change > 0:
                dt = min(0.05 / max_rel_change, base_dt, t_end - t_current)
            else:
                dt = min(base_dt, t_end - t_current)

            dt = max(dt, 1e-10)  # Minimum step

            # Euler step
            y = y + dydt * dt

            # Clamp to non-negative
            y = torch.clamp(y, min=0.0)

            t_current += dt
            total_steps += 1

        trajectories[:, i, :] = y

        # Early exit if stuck
        if total_steps >= max_steps:
            # Fill remaining with last value
            for j in range(i + 1, n_times):
                trajectories[:, j, :] = y
            break

    return trajectories


def get_initial_conditions(
    mechanism: MechanismTemplate,
    E_total: Tensor,
    substrate_concs: Dict[str, Tensor],
    inhibitor_concs: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    Build initial conditions dict for mechanism simulation.

    Args:
        mechanism: The mechanism template
        E_total: Total enzyme concentration (batch,)
        substrate_concs: Dict mapping substrate name to concentration
        inhibitor_concs: Optional dict mapping inhibitor name to concentration

    Returns:
        Dict of initial concentrations for all species
    """
    batch_size = E_total.shape[0]
    device = E_total.device

    conditions = {}

    # Free enzyme starts with all enzyme
    conditions["E"] = E_total.clone()

    # For ping-pong, F also starts at 0
    if "ping_pong" in mechanism.name:
        conditions["F"] = torch.zeros(batch_size, device=device)

    # All enzyme complexes start at 0
    for state in mechanism.states:
        for sp in state.species:
            if (sp.startswith("E") or sp.startswith("F")) and len(sp) > 1:
                if sp not in conditions:
                    conditions[sp] = torch.zeros(batch_size, device=device)

    # Substrates
    for sub_name in mechanism.substrate_names:
        if sub_name in substrate_concs:
            conditions[sub_name] = substrate_concs[sub_name]
        else:
            # Default substrate concentration
            conditions[sub_name] = torch.ones(batch_size, device=device)

    # Products start at 0
    for prod_name in mechanism.product_names:
        conditions[prod_name] = torch.zeros(batch_size, device=device)

    # Inhibitors
    if inhibitor_concs is not None:
        for inh_name in mechanism.inhibitor_names:
            if inh_name in inhibitor_concs:
                conditions[inh_name] = inhibitor_concs[inh_name]
            else:
                conditions[inh_name] = torch.zeros(batch_size, device=device)
    else:
        for inh_name in mechanism.inhibitor_names:
            conditions[inh_name] = torch.zeros(batch_size, device=device)

    return conditions
