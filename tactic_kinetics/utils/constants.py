"""
Physical constants for thermodynamic calculations.
"""

# Gas constant in different units
R_J_MOL_K = 8.314462618  # J/(mol·K)
R_KJ_MOL_K = 8.314462618e-3  # kJ/(mol·K)
R_CAL_MOL_K = 1.987204258  # cal/(mol·K)
R_KCAL_MOL_K = 1.987204258e-3  # kcal/(mol·K)

# Boltzmann constant
K_B = 1.380649e-23  # J/K

# Planck constant
H = 6.62607015e-34  # J·s

# Avogadro's number
N_A = 6.02214076e23  # mol^-1

# Standard temperature
T_STANDARD = 298.15  # K (25°C)

# Standard pressure
P_STANDARD = 1e5  # Pa (1 bar)

# Transmission coefficient (typically assumed to be 1)
KAPPA = 1.0

# Pre-exponential factor for Eyring equation: k_B * T / h at 298.15 K
# This gives the frequency factor in s^-1
def eyring_prefactor(T: float) -> float:
    """Calculate the Eyring pre-exponential factor k_B*T/h."""
    return K_B * T / H
