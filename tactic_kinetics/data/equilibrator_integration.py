"""
eQuilibrator integration for thermodynamic priors.

This module provides utilities for accessing thermodynamic data
from eQuilibrator to inform energy priors in TACTIC-Kinetics.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import csv

# Try to import equilibrator-api (optional dependency)
try:
    from equilibrator_api import ComponentContribution, Q_
    EQUILIBRATOR_AVAILABLE = True
except ImportError:
    EQUILIBRATOR_AVAILABLE = False
    ComponentContribution = None
    Q_ = None


class EquilibratorClient:
    """
    Client for accessing eQuilibrator thermodynamic data.

    Provides standard Gibbs energies for reactions and compounds,
    which are used to set priors on energy parameters.
    """

    def __init__(
        self,
        ph: float = 7.0,
        ionic_strength: float = 0.1,
        temperature: float = 298.15,
        pmg: float = 3.0,
    ):
        """
        Args:
            ph: Default pH for calculations
            ionic_strength: Default ionic strength in M
            temperature: Default temperature in K
            pmg: Default pMg (-log10 of Mg2+ concentration)
        """
        self.ph = ph
        self.ionic_strength = ionic_strength
        self.temperature = temperature
        self.pmg = pmg

        if EQUILIBRATOR_AVAILABLE:
            print("Initializing eQuilibrator (this may take a moment)...")
            self.cc = ComponentContribution()
            self._set_conditions()
        else:
            self.cc = None
            print("Warning: equilibrator-api not available. Using cached data only.")

    def _set_conditions(self):
        """Set thermodynamic conditions."""
        if self.cc is None:
            return

        # Set conditions using Pint quantities
        self.cc.p_h = Q_(self.ph)
        self.cc.ionic_strength = Q_(self.ionic_strength, "M")
        self.cc.temperature = Q_(self.temperature, "K")
        self.cc.p_mg = Q_(self.pmg)

    def get_reaction_dg(
        self,
        reaction_str: str,
        ph: Optional[float] = None,
        ionic_strength: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Get standard Gibbs energy for a reaction.

        Args:
            reaction_str: Reaction string in eQuilibrator format
                         e.g., "kegg:C00002 + kegg:C00001 = kegg:C00008 + kegg:C00009"
            ph: pH (overrides default)
            ionic_strength: Ionic strength in M (overrides default)
            temperature: Temperature in K (overrides default)

        Returns:
            Tuple of (dG_mean, dG_std) in kJ/mol
        """
        if self.cc is None:
            # Return default values if eQuilibrator not available
            return 0.0, 20.0

        # Update conditions if provided
        if ph is not None:
            self.cc.p_h = Q_(ph)
        if ionic_strength is not None:
            self.cc.ionic_strength = Q_(ionic_strength, "M")
        if temperature is not None:
            self.cc.temperature = Q_(temperature, "K")

        try:
            # Parse and calculate
            rxn = self.cc.parse_reaction_formula(reaction_str)

            if not rxn.is_balanced():
                print(f"Warning: Reaction not balanced: {reaction_str}")

            dg_prime = self.cc.standard_dg_prime(rxn)

            # Extract value and uncertainty
            dg_mean = dg_prime.value.magnitude  # kJ/mol
            dg_std = dg_prime.error.magnitude  # kJ/mol

            return float(dg_mean), float(dg_std)

        except Exception as e:
            print(f"Error calculating ΔG° for {reaction_str}: {e}")
            return 0.0, 20.0

    def get_compound_dg_formation(
        self,
        compound_id: str,
    ) -> Tuple[float, float]:
        """
        Get standard Gibbs energy of formation for a compound.

        Args:
            compound_id: Compound identifier (e.g., "kegg:C00002" for ATP)

        Returns:
            Tuple of (dGf_mean, dGf_std) in kJ/mol
        """
        if self.cc is None:
            return 0.0, 20.0

        try:
            compound = self.cc.get_compound(compound_id)
            dgf = self.cc.standard_dg_formation(compound)

            return float(dgf.value.magnitude), float(dgf.error.magnitude)

        except Exception as e:
            print(f"Error getting ΔGf° for {compound_id}: {e}")
            return 0.0, 20.0

    def get_keq(
        self,
        reaction_str: str,
    ) -> float:
        """
        Get equilibrium constant for a reaction.

        Args:
            reaction_str: Reaction string in eQuilibrator format

        Returns:
            Equilibrium constant (dimensionless)
        """
        dg_mean, _ = self.get_reaction_dg(reaction_str)
        R = 8.314462618e-3  # kJ/(mol·K)
        keq = np.exp(-dg_mean / (R * self.temperature))
        return float(keq)


def get_reaction_dg(
    reaction_str: str,
    ph: float = 7.0,
    ionic_strength: float = 0.1,
    temperature: float = 298.15,
) -> Tuple[float, float]:
    """
    Convenience function to get reaction ΔG°.

    Creates a temporary client for one-off calculations.
    """
    client = EquilibratorClient(ph, ionic_strength, temperature)
    return client.get_reaction_dg(reaction_str)


def get_formation_energies(
    compound_ids: List[str],
) -> Dict[str, Tuple[float, float]]:
    """
    Get formation energies for multiple compounds.

    Returns:
        Dict mapping compound ID to (dGf_mean, dGf_std)
    """
    client = EquilibratorClient()
    return {
        cid: client.get_compound_dg_formation(cid)
        for cid in compound_ids
    }


class ThermodynamicPriorDatabase:
    """
    Database of thermodynamic priors from training data.

    Uses the component contribution training data to provide
    empirical priors on Gibbs energies.
    """

    def __init__(self, data_dir: str = "data/equilibrator"):
        self.data_dir = Path(data_dir)
        self.formation_energies = {}
        self.reaction_energies = {}

        self._load_data()

    def _load_data(self):
        """Load thermodynamic data from CSV files."""
        # Load formation energies
        formation_path = self.data_dir / "formation_energies_transformed.csv"
        if formation_path.exists():
            with open(formation_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    compound_id = row.get("compound_id") or row.get("id")
                    if compound_id:
                        try:
                            dg = float(row.get("dG", row.get("dG_f", 0)))
                            self.formation_energies[compound_id] = dg
                        except (ValueError, TypeError):
                            pass

        # Load TECRDB reaction energies
        tecrdb_path = self.data_dir / "TECRDB.csv"
        if tecrdb_path.exists():
            with open(tecrdb_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rxn_id = row.get("reaction_id") or row.get("id")
                    if rxn_id:
                        try:
                            dg = float(row.get("dG", row.get("K_prime", 0)))
                            self.reaction_energies[rxn_id] = dg
                        except (ValueError, TypeError):
                            pass

        print(f"Loaded {len(self.formation_energies)} formation energies")
        print(f"Loaded {len(self.reaction_energies)} reaction energies")

    def get_formation_energy(self, compound_id: str) -> Optional[float]:
        """Get formation energy for a compound."""
        return self.formation_energies.get(compound_id)

    def get_reaction_energy(self, reaction_id: str) -> Optional[float]:
        """Get reaction energy by ID."""
        return self.reaction_energies.get(reaction_id)

    def get_energy_statistics(self) -> Dict[str, float]:
        """Get statistics of energy distributions."""
        formation_values = list(self.formation_energies.values())
        reaction_values = list(self.reaction_energies.values())

        stats = {}

        if formation_values:
            stats["formation_mean"] = np.mean(formation_values)
            stats["formation_std"] = np.std(formation_values)
            stats["formation_min"] = np.min(formation_values)
            stats["formation_max"] = np.max(formation_values)

        if reaction_values:
            stats["reaction_mean"] = np.mean(reaction_values)
            stats["reaction_std"] = np.std(reaction_values)
            stats["reaction_min"] = np.min(reaction_values)
            stats["reaction_max"] = np.max(reaction_values)

        return stats


class EnergyPriorGenerator:
    """
    Generates energy priors for TACTIC-Kinetics training.

    Combines thermodynamic data from eQuilibrator with empirical
    distributions from BRENDA/literature.
    """

    def __init__(
        self,
        use_equilibrator: bool = True,
        data_dir: str = "data/equilibrator",
    ):
        self.use_equilibrator = use_equilibrator and EQUILIBRATOR_AVAILABLE

        if self.use_equilibrator:
            self.client = EquilibratorClient()
        else:
            self.client = None

        self.db = ThermodynamicPriorDatabase(data_dir)

        # Empirical priors from literature
        # These are typical ranges for enzyme kinetics
        self.empirical_priors = {
            # State energies (relative to E+S)
            "G_ES": {"mean": -10.0, "std": 8.0},   # ES complex typically lower energy
            "G_EP": {"mean": -5.0, "std": 10.0},   # EP complex
            "G_EI": {"mean": -15.0, "std": 10.0},  # Inhibitor binding (tight)

            # Barrier energies
            "G_barrier_bind": {"mean": 40.0, "std": 10.0},   # Binding barrier
            "G_barrier_cat": {"mean": 60.0, "std": 15.0},    # Catalytic barrier
            "G_barrier_release": {"mean": 45.0, "std": 10.0}, # Release barrier
        }

    def get_state_energy_prior(
        self,
        state_name: str,
    ) -> Tuple[float, float]:
        """
        Get prior mean and std for a state energy.

        Args:
            state_name: Name of the state (e.g., "G_ES")

        Returns:
            Tuple of (mean, std) in kJ/mol
        """
        if state_name in self.empirical_priors:
            prior = self.empirical_priors[state_name]
            return prior["mean"], prior["std"]

        # Default prior
        return 0.0, 20.0

    def get_barrier_energy_prior(
        self,
        barrier_name: str,
    ) -> Tuple[float, float]:
        """
        Get prior mean and std for a barrier energy.

        Args:
            barrier_name: Name of the barrier (e.g., "G_barrier_cat")

        Returns:
            Tuple of (mean, std) in kJ/mol
        """
        if barrier_name in self.empirical_priors:
            prior = self.empirical_priors[barrier_name]
            return prior["mean"], prior["std"]

        # Default barrier prior (should be positive)
        return 60.0, 15.0

    def get_mechanism_priors(
        self,
        mechanism_name: str,
        state_param_names: List[str],
        barrier_param_names: List[str],
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Get all priors for a mechanism.

        Returns:
            Dict with "state_mean", "state_std", "barrier_mean", "barrier_std"
        """
        state_means = []
        state_stds = []
        for name in state_param_names:
            mean, std = self.get_state_energy_prior(name)
            state_means.append(mean)
            state_stds.append(std)

        barrier_means = []
        barrier_stds = []
        for name in barrier_param_names:
            mean, std = self.get_barrier_energy_prior(name)
            barrier_means.append(mean)
            barrier_stds.append(std)

        return {
            "state_mean": torch.tensor(state_means),
            "state_std": torch.tensor(state_stds),
            "barrier_mean": torch.tensor(barrier_means),
            "barrier_std": torch.tensor(barrier_stds),
        }
