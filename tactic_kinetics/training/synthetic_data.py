"""
Synthetic data generation for TACTIC-Kinetics training.

This module generates synthetic enzyme kinetics data with known
mechanism types and energy parameters for training and validation.

Energy parameters are sampled from thermodynamically-grounded priors:
1. Reaction ΔG° distributions from eQuilibrator/TECRDB
2. Activation energy distributions from BRENDA-derived literature values
"""

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from ..mechanisms.base import MechanismTemplate
from ..mechanisms.templates import get_all_mechanisms, get_mechanism_by_name
from ..models.ode_simulator import ODESimulator, simulate_michaelis_menten
from ..utils.constants import T_STANDARD
from ..utils.thermodynamics import eyring_rate_constant
from .thermodynamic_priors import ThermodynamicPriorExtractor, get_thermodynamic_priors


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    # Mechanism selection
    mechanism_names: Optional[List[str]] = None  # None = all mechanisms

    # Thermodynamic priors (recommended: True)
    use_thermodynamic_priors: bool = True  # Use eQuilibrator/BRENDA-derived distributions
    equilibrator_data_dir: str = "data/equilibrator"  # Path to eQuilibrator data

    # Fallback energy sampling (only used if use_thermodynamic_priors=False)
    state_energy_mean: float = 0.0  # kJ/mol
    state_energy_std: float = 15.0  # kJ/mol
    barrier_energy_mean: float = 60.0  # kJ/mol
    barrier_energy_std: float = 10.0  # kJ/mol
    min_barrier: float = 20.0  # Minimum barrier height
    max_barrier: float = 100.0  # Maximum barrier height

    # Experimental conditions
    temperature_range: Tuple[float, float] = (293.15, 318.15)  # 20-45°C
    ph_range: Tuple[float, float] = (6.0, 8.0)
    substrate_range: Tuple[float, float] = (0.01, 10.0)  # mM
    enzyme_range: Tuple[float, float] = (0.001, 0.1)  # mM

    # Time series
    t_max: float = 100.0  # Maximum time (seconds)
    n_timepoints: int = 50  # Number of time points per curve
    n_observations: int = 20  # Number of observed points (sparse)

    # Noise
    noise_std: float = 0.02  # Relative noise level
    missing_fraction: float = 0.1  # Fraction of missing observations

    # Dataset size
    n_samples_per_mechanism: int = 1000


class SyntheticDataGenerator:
    """
    Generator for synthetic enzyme kinetics data.

    Generates progress curves with known mechanism types and energy
    parameters for supervised training.

    Energy parameters are sampled from:
    1. eQuilibrator/TECRDB distributions for reaction ΔG° and binding energies
    2. BRENDA-derived distributions for activation energies (barriers)
    """

    def __init__(
        self,
        config: Optional[SyntheticDataConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or SyntheticDataConfig()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Get mechanisms
        if self.config.mechanism_names is None:
            self.mechanisms = get_all_mechanisms()
        else:
            self.mechanisms = {
                name: get_mechanism_by_name(name)
                for name in self.config.mechanism_names
            }

        self.mechanism_names = list(self.mechanisms.keys())
        self.n_mechanisms = len(self.mechanism_names)

        # Load thermodynamic priors if enabled
        self.thermo_priors = None
        if self.config.use_thermodynamic_priors:
            try:
                self.thermo_priors = get_thermodynamic_priors(
                    self.config.equilibrator_data_dir
                )
                print("Using thermodynamically-grounded priors from eQuilibrator/BRENDA")
            except Exception as e:
                print(f"Warning: Could not load thermodynamic priors: {e}")
                print("Falling back to default Gaussian sampling")

    def sample_energies(
        self,
        mechanism: MechanismTemplate,
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample energy parameters for a mechanism from thermodynamic priors.

        When use_thermodynamic_priors=True (default), samples from:
        - eQuilibrator/TECRDB distributions for state energies (ΔG°)
        - BRENDA-derived distributions for barrier energies (ΔG‡)

        Returns:
            Tuple of (state_energies, barrier_energies)
        """
        n_states = mechanism.n_energy_params
        n_barriers = mechanism.n_barrier_params

        if self.thermo_priors is not None:
            # Use thermodynamically-grounded priors
            state_energies = self._sample_state_energies_from_priors(
                mechanism, batch_size
            )
            barrier_energies = self._sample_barrier_energies_from_priors(
                mechanism, batch_size
            )
        else:
            # Fallback to simple Gaussian sampling
            state_energies = torch.randn(batch_size, n_states) * self.config.state_energy_std
            state_energies = state_energies + self.config.state_energy_mean

            barrier_energies = torch.randn(batch_size, n_barriers) * self.config.barrier_energy_std
            barrier_energies = barrier_energies + self.config.barrier_energy_mean
            barrier_energies = barrier_energies.clamp(
                self.config.min_barrier,
                self.config.max_barrier,
            )

        return state_energies, barrier_energies

    def _sample_state_energies_from_priors(
        self,
        mechanism: MechanismTemplate,
        batch_size: int,
    ) -> Tensor:
        """
        Sample state energies from eQuilibrator-derived distributions.

        State energies include:
        - E (free enzyme): reference = 0
        - ES (enzyme-substrate complex): binding ΔG from literature
        - EP (enzyme-product complex): binding ΔG
        - Other intermediates: reaction ΔG from TECRDB

        The energies are sampled to be thermodynamically consistent.
        """
        n_states = mechanism.n_energy_params
        state_energies = np.zeros((batch_size, n_states))

        # Get distributions
        binding_dist = self.thermo_priors.get_binding_dg_prior()
        reaction_dist = self.thermo_priors.get_reaction_dg_prior()

        for i in range(n_states):
            # Determine which distribution to use based on state type
            # First state (E) is typically reference = 0
            if i == 0:
                state_energies[:, i] = 0.0
            elif i == 1:
                # ES complex: sample binding energy
                state_energies[:, i] = binding_dist.sample(batch_size, self.rng)
            else:
                # Other states: use reaction ΔG distribution
                # Relative to previous state
                delta_g = reaction_dist.sample(batch_size, self.rng)
                state_energies[:, i] = state_energies[:, i-1] + delta_g * 0.3

        return torch.tensor(state_energies, dtype=torch.float32)

    def _sample_barrier_energies_from_priors(
        self,
        mechanism: MechanismTemplate,
        batch_size: int,
    ) -> Tensor:
        """
        Sample barrier energies from BRENDA-derived distributions.

        Different transition types have different barrier distributions:
        - Substrate binding: ~40 kJ/mol (fast, diffusion-influenced)
        - Catalysis: ~65 kJ/mol (rate-limiting step)
        - Product release: ~45 kJ/mol (intermediate)
        - Inhibitor binding: ~35 kJ/mol (often tight binding)
        """
        n_barriers = mechanism.n_barrier_params
        barrier_energies = np.zeros((batch_size, n_barriers))

        # Get distributions for different transition types
        catalysis_dist = self.thermo_priors.get_activation_energy_prior('catalysis')
        binding_dist = self.thermo_priors.get_activation_energy_prior('binding')
        release_dist = self.thermo_priors.get_activation_energy_prior('release')
        inhibitor_dist = self.thermo_priors.get_activation_energy_prior('inhibitor_binding')

        # Assign appropriate distribution based on transition index
        # Typical order: binding, catalysis, release
        for i in range(n_barriers):
            if i == 0:
                # First barrier: substrate binding
                barrier_energies[:, i] = binding_dist.sample(batch_size, self.rng)
            elif i == n_barriers - 1:
                # Last barrier: product release
                barrier_energies[:, i] = release_dist.sample(batch_size, self.rng)
            elif 'inhibition' in mechanism.name.lower() and i >= n_barriers // 2:
                # Inhibitor-related barriers
                barrier_energies[:, i] = inhibitor_dist.sample(batch_size, self.rng)
            else:
                # Middle barriers: catalysis
                barrier_energies[:, i] = catalysis_dist.sample(batch_size, self.rng)

        return torch.tensor(barrier_energies, dtype=torch.float32)

    def sample_conditions(self, batch_size: int = 1) -> Tensor:
        """
        Sample experimental conditions.

        Returns:
            Conditions tensor of shape (batch_size, 4)
            Columns: [temperature, pH, [S]_0, [E]_0]
        """
        T = torch.rand(batch_size) * (
            self.config.temperature_range[1] - self.config.temperature_range[0]
        ) + self.config.temperature_range[0]

        pH = torch.rand(batch_size) * (
            self.config.ph_range[1] - self.config.ph_range[0]
        ) + self.config.ph_range[0]

        # Log-uniform sampling for concentrations
        log_S = torch.rand(batch_size) * (
            np.log10(self.config.substrate_range[1]) -
            np.log10(self.config.substrate_range[0])
        ) + np.log10(self.config.substrate_range[0])
        S0 = 10 ** log_S

        log_E = torch.rand(batch_size) * (
            np.log10(self.config.enzyme_range[1]) -
            np.log10(self.config.enzyme_range[0])
        ) + np.log10(self.config.enzyme_range[0])
        E0 = 10 ** log_E

        return torch.stack([T, pH, S0, E0], dim=-1)

    def generate_trajectory_simple(
        self,
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generate simple Michaelis-Menten trajectories.

        This is a fast method using analytical/semi-analytical solutions.
        Energy parameters are sampled from thermodynamic priors when available.

        Returns:
            Tuple of (times, concentrations, conditions, kinetic_params)
        """
        # Sample conditions
        conditions = self.sample_conditions(batch_size)
        T = conditions[:, 0]
        S0 = conditions[:, 2]
        E0 = conditions[:, 3]

        # Sample kinetic parameters via energies
        if self.thermo_priors is not None:
            # Use thermodynamically-grounded priors
            binding_dist = self.thermo_priors.get_binding_dg_prior()
            catalysis_dist = self.thermo_priors.get_activation_energy_prior('catalysis')

            # G_ES: binding energy from eQuilibrator-derived distribution
            G_ES = torch.tensor(
                binding_dist.sample(batch_size, self.rng),
                dtype=torch.float32
            )
            # G_barrier: activation energy from BRENDA-derived distribution
            G_barrier = torch.tensor(
                catalysis_dist.sample(batch_size, self.rng),
                dtype=torch.float32
            )
        else:
            # Fallback to random Gaussian
            G_ES = torch.randn(batch_size) * 10.0 - 5.0  # Binding energy
            G_barrier = torch.randn(batch_size) * 10.0 + 60.0  # Activation energy
            G_barrier = G_barrier.clamp(30.0, 90.0)

        # Convert to kcat and Km
        # kcat = (kB*T/h) * exp(-G_barrier / RT)
        from ..utils.constants import K_B, H, R_KJ_MOL_K
        prefactor = K_B * T / H
        RT = R_KJ_MOL_K * T

        kcat = prefactor * torch.exp(-G_barrier / RT)

        # Km depends on binding energy
        # Kd = exp(G_ES / RT) is the dissociation constant
        # Km ≈ Kd for rapid equilibrium assumption
        # Scale to typical Km range (0.01-10 mM)
        Kd = torch.exp(G_ES / RT)
        Km = Kd * 1.0  # Scale factor for mM units

        # Clamp to numerically stable range for ODE solver
        kcat = kcat.clamp(1e-4, 1e6)  # s^-1
        Km = Km.clamp(0.001, 100.0)  # mM

        # Generate time points
        t_eval = torch.linspace(0, self.config.t_max, self.config.n_timepoints)

        # Simulate
        S, P = simulate_michaelis_menten(kcat, Km, E0, S0, t_eval)

        # Add noise
        if self.config.noise_std > 0:
            noise = torch.randn_like(P) * self.config.noise_std * P.abs().mean(dim=-1, keepdim=True)
            P = P + noise

        # Stack kinetic params
        kinetic_params = torch.stack([kcat, Km, G_ES, G_barrier], dim=-1)

        # Expand times for batch
        times = t_eval.unsqueeze(0).expand(batch_size, -1)

        return times, P, conditions, kinetic_params

    def generate_sample(
        self,
        mechanism_name: str,
    ) -> Dict[str, Tensor]:
        """
        Generate a single sample for a specific mechanism.

        Returns:
            Dict containing all sample data
        """
        mechanism = self.mechanisms[mechanism_name]

        # Sample energies
        state_energies, barrier_energies = self.sample_energies(mechanism, batch_size=1)

        # Sample conditions
        conditions = self.sample_conditions(batch_size=1)

        # For now, use simplified simulation
        # Full mechanism simulation would require the ODE simulator
        times, values, _, _ = self.generate_trajectory_simple(batch_size=1)

        # Sparse observations
        n_obs = self.config.n_observations
        obs_idx = torch.randperm(self.config.n_timepoints)[:n_obs].sort()[0]

        obs_times = times[0, obs_idx]
        obs_values = values[0, obs_idx]

        # Create mask (some observations might be "missing")
        mask = torch.rand(n_obs) > self.config.missing_fraction

        return {
            "mechanism_name": mechanism_name,
            "mechanism_idx": self.mechanism_names.index(mechanism_name),
            "state_energies": state_energies.squeeze(0),
            "barrier_energies": barrier_energies.squeeze(0),
            "conditions": conditions.squeeze(0),
            "times": obs_times,
            "values": obs_values,
            "mask": mask,
            "full_times": times.squeeze(0),
            "full_trajectory": values.squeeze(0),
        }

    def generate_dataset(
        self,
        n_samples_per_mechanism: Optional[int] = None,
    ) -> List[Dict[str, Tensor]]:
        """
        Generate full dataset with samples from all mechanisms.

        Returns:
            List of sample dicts
        """
        if n_samples_per_mechanism is None:
            n_samples_per_mechanism = self.config.n_samples_per_mechanism

        samples = []
        for mech_name in self.mechanism_names:
            for _ in range(n_samples_per_mechanism):
                sample = self.generate_sample(mech_name)
                samples.append(sample)

        return samples


class SyntheticKineticsDataset(Dataset):
    """
    PyTorch Dataset for synthetic enzyme kinetics data.
    """

    def __init__(
        self,
        samples: List[Dict[str, Tensor]],
        max_obs: int = 50,
    ):
        self.samples = samples
        self.max_obs = max_obs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.samples[idx]

        # Pad observations to max_obs
        n_obs = len(sample["times"])
        pad_len = self.max_obs - n_obs

        if pad_len > 0:
            times = torch.cat([sample["times"], torch.zeros(pad_len)])
            values = torch.cat([sample["values"], torch.zeros(pad_len)])
            mask = torch.cat([sample["mask"], torch.zeros(pad_len, dtype=torch.bool)])
        else:
            times = sample["times"][:self.max_obs]
            values = sample["values"][:self.max_obs]
            mask = sample["mask"][:self.max_obs]

        return {
            "times": times,
            "values": values,
            "mask": mask,
            "conditions": sample["conditions"],
            "mechanism_idx": torch.tensor(sample["mechanism_idx"]),
            "state_energies": sample["state_energies"],
            "barrier_energies": sample["barrier_energies"],
        }


def generate_mechanism_dataset(
    mechanism_names: Optional[List[str]] = None,
    n_samples_per_mechanism: int = 1000,
    config: Optional[SyntheticDataConfig] = None,
    seed: Optional[int] = 42,
) -> Tuple[SyntheticKineticsDataset, SyntheticKineticsDataset]:
    """
    Convenience function to generate train/val datasets.

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if config is None:
        config = SyntheticDataConfig(
            mechanism_names=mechanism_names,
            n_samples_per_mechanism=n_samples_per_mechanism,
        )

    generator = SyntheticDataGenerator(config, seed=seed)

    # Generate samples
    all_samples = generator.generate_dataset()

    # Split 80/20
    n_train = int(len(all_samples) * 0.8)
    np.random.shuffle(all_samples)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    train_dataset = SyntheticKineticsDataset(train_samples)
    val_dataset = SyntheticKineticsDataset(val_samples)

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
