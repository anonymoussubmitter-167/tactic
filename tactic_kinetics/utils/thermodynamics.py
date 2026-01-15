"""
Thermodynamic utility functions for enzyme kinetics.

This module provides functions for converting between Gibbs energies and
rate/equilibrium constants, implementing the Eyring equation, and handling
thermodynamic transformations.
"""

import torch
import numpy as np
from typing import Union, Optional

from .constants import R_KJ_MOL_K, R_J_MOL_K, K_B, H, KAPPA, T_STANDARD


# Type alias for numeric types
Numeric = Union[float, np.ndarray, torch.Tensor]


def gibbs_to_keq(
    delta_g: Numeric,
    temperature: float = T_STANDARD,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Convert standard Gibbs energy to equilibrium constant.

    ΔG° = -RT ln(K_eq)
    K_eq = exp(-ΔG° / RT)

    Args:
        delta_g: Standard Gibbs energy change
        temperature: Temperature in Kelvin
        units: Units of delta_g ("kJ/mol" or "J/mol")

    Returns:
        Equilibrium constant (dimensionless)
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K

    if isinstance(delta_g, torch.Tensor):
        return torch.exp(-delta_g / (R * temperature))
    else:
        return np.exp(-delta_g / (R * temperature))


def keq_to_gibbs(
    keq: Numeric,
    temperature: float = T_STANDARD,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Convert equilibrium constant to standard Gibbs energy.

    ΔG° = -RT ln(K_eq)

    Args:
        keq: Equilibrium constant (dimensionless)
        temperature: Temperature in Kelvin
        units: Desired units ("kJ/mol" or "J/mol")

    Returns:
        Standard Gibbs energy change
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K

    if isinstance(keq, torch.Tensor):
        return -R * temperature * torch.log(keq)
    else:
        return -R * temperature * np.log(keq)


def eyring_rate_constant(
    delta_g_barrier: Numeric,
    temperature: float = T_STANDARD,
    kappa: float = KAPPA,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Calculate rate constant from activation Gibbs energy using Eyring equation.

    k = κ * (k_B * T / h) * exp(-ΔG‡ / RT)

    Args:
        delta_g_barrier: Activation Gibbs energy (ΔG‡)
        temperature: Temperature in Kelvin
        kappa: Transmission coefficient (default 1.0)
        units: Units of delta_g_barrier ("kJ/mol" or "J/mol")

    Returns:
        Rate constant in s^-1
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K
    prefactor = kappa * K_B * temperature / H

    if isinstance(delta_g_barrier, torch.Tensor):
        return prefactor * torch.exp(-delta_g_barrier / (R * temperature))
    else:
        return prefactor * np.exp(-delta_g_barrier / (R * temperature))


def rate_constant_to_activation_energy(
    k: Numeric,
    temperature: float = T_STANDARD,
    kappa: float = KAPPA,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Calculate activation Gibbs energy from rate constant (inverse Eyring).

    ΔG‡ = -RT * ln(k * h / (κ * k_B * T))

    Args:
        k: Rate constant in s^-1
        temperature: Temperature in Kelvin
        kappa: Transmission coefficient (default 1.0)
        units: Desired units ("kJ/mol" or "J/mol")

    Returns:
        Activation Gibbs energy
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K
    prefactor = kappa * K_B * temperature / H

    if isinstance(k, torch.Tensor):
        return -R * temperature * torch.log(k / prefactor)
    else:
        return -R * temperature * np.log(k / prefactor)


def arrhenius_rate_constant(
    activation_energy: Numeric,
    pre_exponential: float,
    temperature: float = T_STANDARD,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Calculate rate constant using Arrhenius equation.

    k = A * exp(-E_a / RT)

    Args:
        activation_energy: Activation energy (E_a)
        pre_exponential: Pre-exponential factor (A) in s^-1
        temperature: Temperature in Kelvin
        units: Units of activation_energy ("kJ/mol" or "J/mol")

    Returns:
        Rate constant in s^-1
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K

    if isinstance(activation_energy, torch.Tensor):
        return pre_exponential * torch.exp(-activation_energy / (R * temperature))
    else:
        return pre_exponential * np.exp(-activation_energy / (R * temperature))


def transform_gibbs_ph(
    delta_g_standard: Numeric,
    n_h: int,
    ph: float,
    ph_reference: float = 7.0,
    temperature: float = T_STANDARD,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Transform standard Gibbs energy for pH change.

    ΔG'° = ΔG° + n_H * RT * ln(10) * (pH - pH_ref)

    where n_H is the net number of protons consumed in the reaction.

    Args:
        delta_g_standard: Standard Gibbs energy at reference pH
        n_h: Net protons consumed (positive if consumed, negative if produced)
        ph: Target pH
        ph_reference: Reference pH (default 7.0)
        temperature: Temperature in Kelvin
        units: Units ("kJ/mol" or "J/mol")

    Returns:
        Transformed Gibbs energy at target pH
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K
    ln10 = 2.302585093

    correction = n_h * R * temperature * ln10 * (ph - ph_reference)
    return delta_g_standard + correction


def transform_gibbs_ionic_strength(
    delta_g_standard: Numeric,
    charge_squared_change: float,
    ionic_strength: float,
    temperature: float = T_STANDARD,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Transform standard Gibbs energy for ionic strength using Debye-Hückel.

    Uses the extended Debye-Hückel equation for ionic strength correction.

    ΔG'° = ΔG° - 2.91482 * Δ(z²) * sqrt(I) / (1 + 1.6 * sqrt(I))

    where Δ(z²) is the change in sum of squared charges and I is ionic strength.

    Args:
        delta_g_standard: Standard Gibbs energy at zero ionic strength
        charge_squared_change: Change in sum of squared charges (products - reactants)
        ionic_strength: Ionic strength in M
        temperature: Temperature in Kelvin
        units: Units ("kJ/mol" or "J/mol")

    Returns:
        Transformed Gibbs energy at given ionic strength
    """
    # Debye-Hückel constant at 298.15 K (in kJ/mol for z=1)
    A_DH = 2.91482  # kJ/mol at 298.15 K

    if units == "J/mol":
        A_DH *= 1000

    # Temperature correction (approximate)
    A_DH *= (temperature / T_STANDARD) ** 1.5

    sqrt_I = ionic_strength ** 0.5 if not isinstance(ionic_strength, torch.Tensor) else torch.sqrt(torch.tensor(ionic_strength))

    correction = -A_DH * charge_squared_change * sqrt_I / (1 + 1.6 * sqrt_I)
    return delta_g_standard + correction


def binding_gibbs_to_kd(
    delta_g_bind: Numeric,
    temperature: float = T_STANDARD,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Convert binding Gibbs energy to dissociation constant.

    ΔG_bind = RT * ln(K_d)
    K_d = exp(ΔG_bind / RT)

    Note: More negative ΔG_bind means tighter binding (smaller K_d).

    Args:
        delta_g_bind: Binding Gibbs energy (negative for favorable binding)
        temperature: Temperature in Kelvin
        units: Units of delta_g_bind ("kJ/mol" or "J/mol")

    Returns:
        Dissociation constant in M
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K

    if isinstance(delta_g_bind, torch.Tensor):
        return torch.exp(delta_g_bind / (R * temperature))
    else:
        return np.exp(delta_g_bind / (R * temperature))


def kd_to_binding_gibbs(
    kd: Numeric,
    temperature: float = T_STANDARD,
    units: str = "kJ/mol"
) -> Numeric:
    """
    Convert dissociation constant to binding Gibbs energy.

    ΔG_bind = RT * ln(K_d)

    Args:
        kd: Dissociation constant in M
        temperature: Temperature in Kelvin
        units: Desired units ("kJ/mol" or "J/mol")

    Returns:
        Binding Gibbs energy
    """
    R = R_KJ_MOL_K if units == "kJ/mol" else R_J_MOL_K

    if isinstance(kd, torch.Tensor):
        return R * temperature * torch.log(kd)
    else:
        return R * temperature * np.log(kd)


class ThermodynamicState:
    """
    Container for thermodynamic conditions.

    Attributes:
        temperature: Temperature in Kelvin
        ph: pH value
        ionic_strength: Ionic strength in M
        pmg: pMg value (-log10 of Mg2+ concentration)
    """

    def __init__(
        self,
        temperature: float = T_STANDARD,
        ph: float = 7.0,
        ionic_strength: float = 0.1,
        pmg: float = 3.0
    ):
        self.temperature = temperature
        self.ph = ph
        self.ionic_strength = ionic_strength
        self.pmg = pmg

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "ph": self.ph,
            "ionic_strength": self.ionic_strength,
            "pmg": self.pmg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThermodynamicState":
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"ThermodynamicState(T={self.temperature:.2f}K, "
            f"pH={self.ph:.2f}, I={self.ionic_strength:.3f}M, pMg={self.pmg:.2f})"
        )
