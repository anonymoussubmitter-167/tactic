"""
Multi-condition data generator for TACTIC-Kinetics.

Key insight: one sample = one enzyme measured under multiple conditions.
This enables discrimination of mechanisms that are indistinguishable
from single curves (e.g., competitive vs uncompetitive inhibition).
"""

import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.integrate import odeint
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path

from ..utils.constants import K_B, H, R_KJ_MOL_K, T_STANDARD


@dataclass
class MultiConditionSample:
    """
    A sample consists of multiple trajectories from the SAME enzyme
    measured under different experimental conditions.
    """
    mechanism: str
    mechanism_idx: int
    energy_params: Dict[str, float]  # True underlying parameters (same for all trajectories)
    trajectories: List[Dict] = field(default_factory=list)

    def add_trajectory(self, conditions: Dict, t: np.ndarray, concentrations: Dict[str, np.ndarray]):
        self.trajectories.append({
            'conditions': conditions,
            't': t,
            'concentrations': concentrations
        })

    @property
    def n_conditions(self) -> int:
        return len(self.trajectories)


@dataclass
class MultiConditionConfig:
    """Configuration for multi-condition data generation."""
    n_conditions_per_sample: int = 20  # Increased for better discrimination
    n_timepoints: int = 20
    noise_level: float = 0.03

    # Energy parameter ranges (kJ/mol)
    dG_binding_mean: float = -15.0
    dG_binding_std: float = 8.0
    dG_barrier_mean: float = 60.0
    dG_barrier_std: float = 10.0
    min_barrier: float = 30.0
    max_barrier: float = 90.0

    # Concentration ranges (mM)
    E0_default: float = 1e-3
    S0_range: Tuple[float, float] = (0.01, 10.0)
    I0_range: Tuple[float, float] = (0.001, 1.0)

    # Time range
    t_max_default: float = 100.0


class MultiConditionGenerator:
    """
    Generates multi-condition samples where mechanism discrimination
    is theoretically possible.
    """

    MECHANISMS = [
        'michaelis_menten_irreversible',
        'michaelis_menten_reversible',
        'competitive_inhibition',
        'uncompetitive_inhibition',
        'mixed_inhibition',
        'substrate_inhibition',
        'ordered_bi_bi',
        'random_bi_bi',
        'ping_pong',
        'product_inhibition',
    ]

    def __init__(self,
                 config: Optional[MultiConditionConfig] = None,
                 seed: Optional[int] = None):
        self.config = config or MultiConditionConfig()
        self.seed_base = seed

        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Mechanism-specific condition variation strategies
        self.condition_strategies = {
            'michaelis_menten_irreversible': self._vary_substrate,
            'michaelis_menten_reversible': self._vary_substrate_with_product,
            'competitive_inhibition': self._vary_inhibitor,
            'uncompetitive_inhibition': self._vary_inhibitor,
            'mixed_inhibition': self._vary_inhibitor,
            'substrate_inhibition': self._vary_substrate_high,
            'ordered_bi_bi': self._vary_both_substrates,
            'random_bi_bi': self._vary_both_substrates,
            'ping_pong': self._vary_both_substrates,
            'product_inhibition': self._vary_substrate_with_product,
        }

    def generate_sample(self, mechanism: str) -> MultiConditionSample:
        """Generate a multi-condition sample for a given mechanism."""

        mechanism_idx = self.MECHANISMS.index(mechanism)

        # Sample thermodynamic parameters (FIXED for this enzyme)
        energy_params = self._sample_energy_params(mechanism)

        sample = MultiConditionSample(
            mechanism=mechanism,
            mechanism_idx=mechanism_idx,
            energy_params=energy_params
        )

        # Get condition variation strategy for this mechanism
        strategy = self.condition_strategies[mechanism]
        conditions_list = strategy(energy_params)

        # Simulate each condition
        for conditions in conditions_list:
            try:
                t, conc = self._simulate(mechanism, energy_params, conditions)

                # Add noise
                conc_noisy = self._add_noise(conc)

                sample.add_trajectory(conditions, t, conc_noisy)
            except Exception as e:
                # If simulation fails, skip this condition
                continue

        return sample

    def generate_batch(self, n_samples_per_mechanism: int, n_workers: int = None) -> List[MultiConditionSample]:
        """
        Generate a batch of samples for all mechanisms using multiprocessing.

        Args:
            n_samples_per_mechanism: Number of samples per mechanism
            n_workers: Number of parallel workers (default: all CPU cores, min 1)
        """
        if n_workers is None:
            n_workers = cpu_count()

        # Build list of (mechanism, seed) tuples for parallel generation
        tasks = []
        for mech_idx, mechanism in enumerate(self.MECHANISMS):
            for i in range(n_samples_per_mechanism):
                # Unique seed for each sample
                seed = (self.seed_base + mech_idx * 100000 + i) if self.seed_base else None
                tasks.append((mechanism, seed))

        total_samples = len(tasks)

        # Only use multiprocessing if we have enough samples to benefit
        # Overhead makes it slower for small batches
        use_parallel = n_workers > 1 and total_samples >= 500

        if use_parallel:
            print(f"Generating {total_samples} samples using {n_workers} CPU cores (parallel)...")
            with Pool(n_workers) as pool:
                results = pool.starmap(
                    _generate_single_sample_worker,
                    [(self.config, mech, seed) for mech, seed in tasks]
                )
        else:
            print(f"Generating {total_samples} samples (sequential)...")
            results = []
            for i, (mech, seed) in enumerate(tasks):
                if i % 100 == 0 and i > 0:
                    print(f"  Progress: {i}/{total_samples}")
                results.append(_generate_single_sample_worker(self.config, mech, seed))

        # Filter out failed samples
        samples = [s for s in results if s is not None and s.n_conditions >= 3]
        print(f"Generated {len(samples)} valid samples ({total_samples - len(samples)} failed/filtered)")

        return samples

    def generate_batch_sequential(self, n_samples_per_mechanism: int) -> List[MultiConditionSample]:
        """Generate a batch of samples sequentially (for debugging)."""
        samples = []
        for mechanism in self.MECHANISMS:
            for _ in range(n_samples_per_mechanism):
                sample = self.generate_sample(mechanism)
                if sample.n_conditions >= 3:
                    samples.append(sample)
        return samples

    # ========== ENERGY PARAMETER SAMPLING ==========

    def _sample_energy_params(self, mechanism: str) -> Dict[str, float]:
        """Sample thermodynamic parameters for a mechanism."""
        params = {}

        # Binding energies (negative = favorable binding)
        params['dG_ES'] = self.rng.normal(
            self.config.dG_binding_mean,
            self.config.dG_binding_std
        )

        # Barrier energy (always positive)
        params['dG_barrier'] = np.clip(
            self.rng.normal(self.config.dG_barrier_mean, self.config.dG_barrier_std),
            self.config.min_barrier,
            self.config.max_barrier
        )

        # Mechanism-specific parameters
        if mechanism in ['competitive_inhibition', 'uncompetitive_inhibition', 'mixed_inhibition']:
            params['dG_EI'] = self.rng.normal(self.config.dG_binding_mean, self.config.dG_binding_std)
            if mechanism == 'mixed_inhibition':
                params['dG_ESI'] = self.rng.normal(self.config.dG_binding_mean, self.config.dG_binding_std)

        elif mechanism == 'substrate_inhibition':
            params['dG_ESS'] = self.rng.normal(-5.0, 5.0)  # Weaker binding for inhibitory site

        elif mechanism in ['ordered_bi_bi', 'random_bi_bi', 'ping_pong']:
            params['dG_EA'] = self.rng.normal(self.config.dG_binding_mean, self.config.dG_binding_std)
            params['dG_EB'] = self.rng.normal(self.config.dG_binding_mean, self.config.dG_binding_std)
            params['dG_barrier_2'] = np.clip(
                self.rng.normal(self.config.dG_barrier_mean, self.config.dG_barrier_std),
                self.config.min_barrier,
                self.config.max_barrier
            )

        elif mechanism in ['michaelis_menten_reversible', 'product_inhibition']:
            params['dG_EP'] = self.rng.normal(self.config.dG_binding_mean + 5, self.config.dG_binding_std)
            params['dG_barrier_rev'] = np.clip(
                self.rng.normal(self.config.dG_barrier_mean + 10, self.config.dG_barrier_std),
                self.config.min_barrier,
                self.config.max_barrier
            )

        return params

    def _energy_to_km(self, dG: float, T: float = T_STANDARD) -> float:
        """Convert binding energy to Km (mM)."""
        RT = R_KJ_MOL_K * T
        Kd = np.exp(dG / RT)  # Dissociation constant
        return np.clip(Kd, 0.001, 100.0)

    def _energy_to_ki(self, dG: float, T: float = T_STANDARD) -> float:
        """Convert inhibitor binding energy to Ki (mM)."""
        return self._energy_to_km(dG, T)

    def _energy_to_kcat(self, dG_barrier: float, T: float = T_STANDARD) -> float:
        """Convert barrier energy to kcat (s^-1)."""
        RT = R_KJ_MOL_K * T
        prefactor = K_B * T / H
        kcat = prefactor * np.exp(-dG_barrier / RT)
        return np.clip(kcat, 1e-4, 1e6)

    # ========== CONDITION VARIATION STRATEGIES ==========

    def _vary_substrate(self, params: Dict) -> List[Dict]:
        """For simple MM: vary [S] extensively to see full saturation behavior."""
        Km = self._energy_to_km(params['dG_ES'])

        # More conditions spanning wide range around Km
        S0_values = Km * np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])

        return [
            {'S0': float(S0), 'E0': self.config.E0_default, 'T': T_STANDARD, 'pH': 7.0}
            for S0 in S0_values
        ][:self.config.n_conditions_per_sample]

    def _vary_inhibitor(self, params: Dict) -> List[Dict]:
        """
        IMPROVED inhibitor variation using full [I] × [S] grid.

        Key insight: Need to see BOTH how curves shift with [I]
        AND measure apparent Km at each [I] by varying [S].

        This creates a grid that reveals:
        - Competitive: Km_app ↑↑↑ with [I], Vmax unchanged
        - Uncompetitive: Both Km_app ↓ and Vmax ↓ with [I]
        - Mixed: Both change but with different ratio
        """
        Ki = self._energy_to_ki(params['dG_EI'])
        Km = self._energy_to_km(params['dG_ES'])

        conditions = []

        # Full [I] × [S] grid: 4 [I] levels × 5 [S] levels = 20 conditions
        I_levels = [0.0, 0.5, 1.0, 2.0]  # Relative to Ki
        S_levels = [0.2, 0.5, 1.0, 2.0, 5.0]  # Relative to Km

        for I_rel in I_levels:
            for S_rel in S_levels:
                conditions.append({
                    'S0': float(S_rel * Km),
                    'I0': float(I_rel * Ki),
                    'E0': self.config.E0_default,
                    'T': T_STANDARD,
                    'pH': 7.0
                })

        return conditions[:self.config.n_conditions_per_sample]

    def _vary_substrate_high(self, params: Dict) -> List[Dict]:
        """
        For substrate inhibition: need HIGH [S] to see inhibition.
        The biphasic behavior only appears when [S] >> Ki,S.

        More conditions to clearly see the rise-then-fall pattern.
        """
        Km = self._energy_to_km(params['dG_ES'])
        Ki_S = self._energy_to_ki(params['dG_ESS'])

        # Must go well above Ki,S to see inhibition - more points for clear pattern
        S0_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]) * max(Km, Ki_S)

        return [
            {'S0': float(S0), 'E0': self.config.E0_default, 'T': T_STANDARD, 'pH': 7.0}
            for S0 in S0_values
        ][:self.config.n_conditions_per_sample]

    def _vary_both_substrates(self, params: Dict) -> List[Dict]:
        """
        IMPROVED bi-substrate variation using slices through [A]-[B] space.

        Key insight: To distinguish ordered/random/ping-pong, we need
        to see how varying ONE substrate affects kinetics at FIXED levels
        of the OTHER. This reveals the intersection pattern:

        Ordered bi-bi:  Lines intersect LEFT of y-axis (Ki_A term)
        Random bi-bi:   Lines intersect ON the y-axis
        Ping-pong:      Lines are PARALLEL (no intersection)
        """
        Km_A = self._energy_to_km(params['dG_EA'])
        Km_B = self._energy_to_km(params['dG_EB'])

        conditions = []

        # Slice 1: Fix [B] at LOW level, vary [A] widely (5 conditions)
        B_low = 0.2 * Km_B
        for A_mult in [0.1, 0.5, 1.0, 2.0, 5.0]:
            conditions.append({
                'A0': float(A_mult * Km_A),
                'B0': float(B_low),
                'E0': self.config.E0_default,
                'T': T_STANDARD,
                'pH': 7.0
            })

        # Slice 2: Fix [B] at HIGH level, vary [A] widely (5 conditions)
        B_high = 5.0 * Km_B
        for A_mult in [0.1, 0.5, 1.0, 2.0, 5.0]:
            conditions.append({
                'A0': float(A_mult * Km_A),
                'B0': float(B_high),
                'E0': self.config.E0_default,
                'T': T_STANDARD,
                'pH': 7.0
            })

        # Slice 3: Fix [A] at LOW level, vary [B] widely (5 conditions)
        A_low = 0.2 * Km_A
        for B_mult in [0.1, 0.5, 1.0, 2.0, 5.0]:
            conditions.append({
                'A0': float(A_low),
                'B0': float(B_mult * Km_B),
                'E0': self.config.E0_default,
                'T': T_STANDARD,
                'pH': 7.0
            })

        # Slice 4: Fix [A] at HIGH level, vary [B] widely (5 conditions)
        A_high = 5.0 * Km_A
        for B_mult in [0.1, 0.5, 1.0, 2.0, 5.0]:
            conditions.append({
                'A0': float(A_high),
                'B0': float(B_mult * Km_B),
                'E0': self.config.E0_default,
                'T': T_STANDARD,
                'pH': 7.0
            })

        return conditions[:self.config.n_conditions_per_sample]  # Returns up to 20 conditions

    def _vary_substrate_with_product(self, params: Dict) -> List[Dict]:
        """
        IMPROVED reversible/product inhibition variation.

        Key distinction:
        - Reversible: Approaches equilibrium (net rate → 0 when [P]/[S] = Keq)
        - Product inhibition: Rate decreases but reaction continues forward

        Strategy: Full [S] × [P] grid to see equilibrium vs inhibition behavior
        """
        Km = self._energy_to_km(params['dG_ES'])
        Km_P = self._energy_to_km(params['dG_EP'])

        conditions = []

        # Full [S] × [P] grid: 4 [S] levels × 4 [P] levels = 16 conditions
        S_levels = [0.2, 0.5, 1.0, 3.0]  # Relative to Km
        P_levels = [0.0, 0.5, 1.0, 2.0]  # Relative to Km_P

        for S_rel in S_levels:
            for P_rel in P_levels:
                conditions.append({
                    'S0': float(S_rel * Km),
                    'P0': float(P_rel * Km_P),
                    'E0': self.config.E0_default,
                    'T': T_STANDARD,
                    'pH': 7.0
                })

        # Additional conditions near equilibrium (for reversible detection)
        # High [P]/[S] ratio to see approach to equilibrium
        for ratio in [1.0, 2.0, 5.0]:
            conditions.append({
                'S0': float(Km),
                'P0': float(ratio * Km),
                'E0': self.config.E0_default,
                'T': T_STANDARD,
                'pH': 7.0
            })

        return conditions[:self.config.n_conditions_per_sample]

    # ========== SIMULATION ==========

    def _simulate(self, mechanism: str, params: Dict, conditions: Dict) -> Tuple[np.ndarray, Dict]:
        """Simulate ODE system for given mechanism and conditions."""

        T = conditions.get('T', T_STANDARD)

        # Convert energies to rate constants
        kcat = self._energy_to_kcat(params['dG_barrier'], T)
        Km = self._energy_to_km(params['dG_ES'], T)

        rates = {'kcat': kcat, 'Km': Km}

        # Add mechanism-specific rates
        if mechanism in ['competitive_inhibition', 'uncompetitive_inhibition', 'mixed_inhibition']:
            rates['Ki'] = self._energy_to_ki(params['dG_EI'], T)
            if mechanism == 'mixed_inhibition':
                rates['Ki_prime'] = self._energy_to_ki(params['dG_ESI'], T)

        elif mechanism == 'substrate_inhibition':
            rates['Ki_S'] = self._energy_to_ki(params['dG_ESS'], T)

        elif mechanism in ['ordered_bi_bi', 'random_bi_bi', 'ping_pong']:
            rates['Km_A'] = self._energy_to_km(params['dG_EA'], T)
            rates['Km_B'] = self._energy_to_km(params['dG_EB'], T)
            rates['Ki_A'] = rates['Km_A'] * 0.5  # Approximate

        elif mechanism in ['michaelis_menten_reversible', 'product_inhibition']:
            rates['Km_P'] = self._energy_to_km(params['dG_EP'], T)
            rates['kcat_rev'] = self._energy_to_kcat(params['dG_barrier_rev'], T)

        # Get ODE system for mechanism
        ode_func, initial_state, species = self._get_ode_system(
            mechanism, rates, conditions
        )

        # Determine appropriate time span
        t_max = self._estimate_reaction_time(rates, conditions)
        t = np.linspace(0, t_max, self.config.n_timepoints)

        # Solve ODE
        solution = odeint(ode_func, initial_state, t, full_output=False)

        # Package concentrations
        concentrations = {
            species[i]: solution[:, i] for i in range(len(species))
        }

        return t, concentrations

    def _get_ode_system(self, mechanism: str, rates: Dict, conditions: Dict):
        """Return ODE function, initial state, and species list for mechanism."""

        E0 = conditions['E0']

        if mechanism == 'michaelis_menten_irreversible':
            S0 = conditions['S0']
            def ode(y, t):
                S = max(y[0], 0)
                v = rates['kcat'] * E0 * S / (rates['Km'] + S)
                return [-v, v]
            return ode, [S0, 0.0], ['S', 'P']

        elif mechanism == 'michaelis_menten_reversible':
            S0 = conditions['S0']
            P0 = conditions.get('P0', 0.0)
            def ode(y, t):
                S, P = max(y[0], 0), max(y[1], 0)
                v_fwd = rates['kcat'] * E0 * S / (rates['Km'] + S)
                v_rev = rates['kcat_rev'] * E0 * P / (rates['Km_P'] + P)
                v_net = v_fwd - v_rev
                return [-v_net, v_net]
            return ode, [S0, P0], ['S', 'P']

        elif mechanism == 'competitive_inhibition':
            S0, I0 = conditions['S0'], conditions['I0']
            def ode(y, t):
                S = max(y[0], 0)
                Km_app = rates['Km'] * (1 + I0 / rates['Ki'])
                v = rates['kcat'] * E0 * S / (Km_app + S)
                return [-v, v]
            return ode, [S0, 0.0], ['S', 'P']

        elif mechanism == 'uncompetitive_inhibition':
            S0, I0 = conditions['S0'], conditions['I0']
            def ode(y, t):
                S = max(y[0], 0)
                alpha_prime = 1 + I0 / rates['Ki']
                Vmax_app = rates['kcat'] * E0 / alpha_prime
                Km_app = rates['Km'] / alpha_prime
                v = Vmax_app * S / (Km_app + S)
                return [-v, v]
            return ode, [S0, 0.0], ['S', 'P']

        elif mechanism == 'mixed_inhibition':
            S0, I0 = conditions['S0'], conditions['I0']
            def ode(y, t):
                S = max(y[0], 0)
                alpha = 1 + I0 / rates['Ki']
                alpha_prime = 1 + I0 / rates['Ki_prime']
                Vmax_app = rates['kcat'] * E0 / alpha_prime
                Km_app = rates['Km'] * alpha / alpha_prime
                v = Vmax_app * S / (Km_app + S)
                return [-v, v]
            return ode, [S0, 0.0], ['S', 'P']

        elif mechanism == 'substrate_inhibition':
            S0 = conditions['S0']
            def ode(y, t):
                S = max(y[0], 0)
                v = (rates['kcat'] * E0 * S /
                     (rates['Km'] + S + S**2 / rates['Ki_S']))
                return [-v, v]
            return ode, [S0, 0.0], ['S', 'P']

        elif mechanism == 'ordered_bi_bi':
            A0, B0 = conditions['A0'], conditions['B0']
            def ode(y, t):
                A, B = max(y[0], 0), max(y[1], 0)
                denom = (rates['Ki_A'] * rates['Km_B'] +
                         rates['Km_B'] * A + rates['Km_A'] * B + A * B)
                v = rates['kcat'] * E0 * A * B / (denom + 1e-10)
                return [-v, -v, v, v]
            return ode, [A0, B0, 0.0, 0.0], ['A', 'B', 'P', 'Q']

        elif mechanism == 'random_bi_bi':
            A0, B0 = conditions['A0'], conditions['B0']
            def ode(y, t):
                A, B = max(y[0], 0), max(y[1], 0)
                denom = (rates['Km_A'] * rates['Km_B'] +
                         rates['Km_B'] * A + rates['Km_A'] * B + A * B)
                v = rates['kcat'] * E0 * A * B / (denom + 1e-10)
                return [-v, -v, v, v]
            return ode, [A0, B0, 0.0, 0.0], ['A', 'B', 'P', 'Q']

        elif mechanism == 'ping_pong':
            A0, B0 = conditions['A0'], conditions['B0']
            def ode(y, t):
                A, B = max(y[0], 0), max(y[1], 0)
                # Ping-pong: parallel lines in double reciprocal
                denom = rates['Km_A'] * B + rates['Km_B'] * A + A * B
                v = rates['kcat'] * E0 * A * B / (denom + 1e-10)
                return [-v, -v, v, v]
            return ode, [A0, B0, 0.0, 0.0], ['A', 'B', 'P', 'Q']

        elif mechanism == 'product_inhibition':
            S0 = conditions['S0']
            P0 = conditions.get('P0', 0.0)
            def ode(y, t):
                S, P = max(y[0], 0), max(y[1], 0)
                # Product inhibits competitively
                Km_app = rates['Km'] * (1 + P / rates['Km_P'])
                v = rates['kcat'] * E0 * S / (Km_app + S)
                return [-v, v]
            return ode, [S0, P0], ['S', 'P']

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

    def _estimate_reaction_time(self, rates: Dict, conditions: Dict) -> float:
        """Estimate appropriate reaction time based on kinetic parameters."""
        kcat = rates['kcat']
        E0 = conditions['E0']

        # Estimate time for ~90% conversion
        # v_max ~ kcat * E0
        # t ~ S0 / v_max for rough estimate
        S0 = conditions.get('S0', conditions.get('A0', 1.0))
        v_max = kcat * E0

        t_est = 3 * S0 / (v_max + 1e-10)

        return np.clip(t_est, 10.0, self.config.t_max_default)

    def _add_noise(self, concentrations: Dict) -> Dict:
        """Add measurement noise to concentrations."""
        noisy = {}
        for species, conc in concentrations.items():
            noise = self.rng.normal(0, self.config.noise_level * np.abs(conc).mean(), size=conc.shape)
            noisy[species] = np.maximum(conc + noise, 0)  # Concentrations can't be negative
        return noisy


# ========== MULTIPROCESSING WORKER FUNCTION ==========

def _generate_single_sample_worker(config: MultiConditionConfig, mechanism: str, seed: int) -> Optional[MultiConditionSample]:
    """
    Worker function for parallel sample generation.
    Must be at module level for multiprocessing to pickle it.
    """
    try:
        # Create a generator with the specific seed
        generator = MultiConditionGenerator(config, seed=seed)
        return generator.generate_sample(mechanism)
    except Exception as e:
        return None


# ========== DATASET SAVE/LOAD UTILITIES ==========

def save_dataset(samples: List[MultiConditionSample], path: str, config: MultiConditionConfig = None):
    """
    Save generated samples to disk.

    Args:
        samples: List of MultiConditionSample objects
        path: Path to save file (.pt)
        config: Optional config to save with dataset
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert samples to serializable format
    serialized = []
    for sample in samples:
        s = {
            'mechanism': sample.mechanism,
            'mechanism_idx': sample.mechanism_idx,
            'energy_params': sample.energy_params,
            'trajectories': sample.trajectories,
        }
        serialized.append(s)

    data = {
        'samples': serialized,
        'n_samples': len(samples),
        'config': config.__dict__ if config else None,
    }

    torch.save(data, path)
    print(f"Saved {len(samples)} samples to {path}")


def load_dataset(path: str) -> Tuple[List[MultiConditionSample], Optional[MultiConditionConfig]]:
    """
    Load generated samples from disk.

    Args:
        path: Path to saved file (.pt)

    Returns:
        Tuple of (samples, config)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = torch.load(path)

    # Reconstruct samples
    samples = []
    for s in data['samples']:
        sample = MultiConditionSample(
            mechanism=s['mechanism'],
            mechanism_idx=s['mechanism_idx'],
            energy_params=s['energy_params'],
            trajectories=s['trajectories'],
        )
        samples.append(sample)

    # Reconstruct config if available
    config = None
    if data.get('config'):
        config = MultiConditionConfig(**data['config'])

    print(f"Loaded {len(samples)} samples from {path}")
    return samples, config


def generate_and_save_dataset(
    path: str,
    n_samples_per_mechanism: int = 1000,
    n_conditions_per_sample: int = 5,
    n_workers: int = None,
    seed: int = 42,
    force_regenerate: bool = False,
) -> List[MultiConditionSample]:
    """
    Generate dataset and save to disk, or load if already exists.

    Args:
        path: Path to save/load dataset
        n_samples_per_mechanism: Samples per mechanism
        n_conditions_per_sample: Conditions per sample
        n_workers: Number of CPU workers (default: all cores)
        seed: Random seed
        force_regenerate: If True, regenerate even if file exists

    Returns:
        List of samples
    """
    path = Path(path)

    if path.exists() and not force_regenerate:
        print(f"Loading existing dataset from {path}")
        samples, _ = load_dataset(path)
        return samples

    print(f"Generating new dataset...")
    config = MultiConditionConfig(
        n_conditions_per_sample=n_conditions_per_sample,
        n_timepoints=20,
        noise_level=0.03,
    )

    generator = MultiConditionGenerator(config, seed=seed)
    samples = generator.generate_batch(n_samples_per_mechanism, n_workers=n_workers)

    save_dataset(samples, path, config)
    return samples
