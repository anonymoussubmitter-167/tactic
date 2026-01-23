"""
BRENDA database loader for enzyme kinetics data.

This module provides tools for loading and extracting kinetic parameters
from the BRENDA database, either via the SOAP API or from downloaded files.

BRENDA contains:
- Km (Michaelis constants)
- kcat (turnover numbers)
- Ki (inhibition constants)
- Temperature optima
- pH optima

The kinetic parameters can be converted to Gibbs energy estimates
using thermodynamic relationships:
- Km relates to substrate binding energy: ΔG_bind ≈ RT ln(Km)
- kcat relates to activation energy via Eyring equation: ΔG‡ = RT ln(kB*T / (h * kcat))
- Ki relates to inhibitor binding energy: ΔG_inhibitor ≈ RT ln(Ki)
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch

# Constants
R = 8.314462618  # J/(mol·K)
R_KJ = 8.314462618e-3  # kJ/(mol·K)
K_B = 1.380649e-23  # J/K
H = 6.62607015e-34  # J·s
T_STANDARD = 298.15  # K


@dataclass
class KineticParameter:
    """A single kinetic parameter entry from BRENDA."""
    ec_number: str
    organism: str
    value: float
    value_max: Optional[float] = None
    substrate: Optional[str] = None
    inhibitor: Optional[str] = None
    temperature: Optional[float] = None
    ph: Optional[float] = None
    commentary: str = ""
    literature: str = ""

    @property
    def has_uncertainty(self) -> bool:
        """Check if a range is provided."""
        return self.value_max is not None and self.value_max != self.value


@dataclass
class EnzymeKinetics:
    """Collection of kinetic parameters for an enzyme."""
    ec_number: str
    organism: Optional[str] = None
    km_values: List[KineticParameter] = field(default_factory=list)
    kcat_values: List[KineticParameter] = field(default_factory=list)
    ki_values: List[KineticParameter] = field(default_factory=list)

    @property
    def has_inhibition_data(self) -> bool:
        """Check if inhibition data is available."""
        return len(self.ki_values) > 0

    def get_best_km(self, substrate: Optional[str] = None) -> Optional[float]:
        """Get the most reliable Km value."""
        km_list = self.km_values
        if substrate:
            km_list = [k for k in km_list if k.substrate and substrate.lower() in k.substrate.lower()]
        if not km_list:
            return None
        # Prefer values without large ranges
        non_range = [k for k in km_list if not k.has_uncertainty]
        if non_range:
            return np.median([k.value for k in non_range])
        return np.median([k.value for k in km_list])

    def get_best_kcat(self, substrate: Optional[str] = None) -> Optional[float]:
        """Get the most reliable kcat value."""
        kcat_list = self.kcat_values
        if substrate:
            kcat_list = [k for k in kcat_list if k.substrate and substrate.lower() in k.substrate.lower()]
        if not kcat_list:
            return None
        non_range = [k for k in kcat_list if not k.has_uncertainty]
        if non_range:
            return np.median([k.value for k in non_range])
        return np.median([k.value for k in kcat_list])

    def get_best_ki(self, inhibitor: Optional[str] = None) -> Optional[float]:
        """Get the most reliable Ki value."""
        ki_list = self.ki_values
        if inhibitor:
            ki_list = [k for k in ki_list if k.inhibitor and inhibitor.lower() in k.inhibitor.lower()]
        if not ki_list:
            return None
        non_range = [k for k in ki_list if not k.has_uncertainty]
        if non_range:
            return np.median([k.value for k in non_range])
        return np.median([k.value for k in ki_list])


def km_to_binding_dg(km_mm: float, temperature: float = T_STANDARD) -> float:
    """
    Convert Km (in mM) to approximate substrate binding ΔG (in kJ/mol).

    Assumes Km ≈ Kd (dissociation constant) for rapid equilibrium.
    ΔG = RT ln(Kd) = RT ln(Km)

    Note: Km is actually (k-1 + kcat) / k1, so this is an approximation.
    For tight binding, Km ≈ Kd.
    """
    # Convert mM to M
    km_m = km_mm * 1e-3
    return R_KJ * temperature * np.log(km_m)


def kcat_to_activation_dg(kcat_s: float, temperature: float = T_STANDARD) -> float:
    """
    Convert kcat (in s^-1) to activation ΔG‡ (in kJ/mol) using Eyring equation.

    k = (kB * T / h) * exp(-ΔG‡ / RT)
    ΔG‡ = RT * ln(kB * T / (h * k))
    """
    prefactor = K_B * temperature / H
    return R_KJ * temperature * np.log(prefactor / kcat_s)


def ki_to_binding_dg(ki_mm: float, temperature: float = T_STANDARD) -> float:
    """
    Convert Ki (in mM) to inhibitor binding ΔG (in kJ/mol).

    Ki is the inhibition constant, approximately equal to Kd for inhibitor.
    ΔG = RT ln(Ki)
    """
    ki_m = ki_mm * 1e-3
    return R_KJ * temperature * np.log(ki_m)


class BRENDALoader:
    """
    Loader for BRENDA enzyme kinetics data.

    Can load data from:
    1. BRENDA SOAP API (requires registration)
    2. Local JSON cache files
    3. Downloaded BRENDA flat files
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize BRENDA loader.

        Args:
            cache_dir: Directory for caching downloaded data
            email: BRENDA account email (for API access)
            password: BRENDA account password (for API access)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/brenda/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.email = email
        self.password = password
        self._client = None

    def _get_client(self):
        """Get or create SOAP client."""
        if self._client is None:
            try:
                from zeep import Client, Settings
                settings = Settings(strict=False)
                self._client = Client(
                    "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl",
                    settings=settings
                )
            except ImportError:
                raise ImportError("zeep package required. Install with: pip install zeep")
        return self._client

    def _get_password_hash(self) -> str:
        """Hash password for BRENDA API."""
        if not self.password:
            raise ValueError("Password required for BRENDA API")
        return hashlib.sha256(self.password.encode("utf-8")).hexdigest()

    def _parse_brenda_result(self, result_string: str) -> List[Dict]:
        """Parse BRENDA result string into list of dictionaries."""
        if not result_string:
            return []

        entries = result_string.split("!")
        parsed = []

        for entry in entries:
            if not entry.strip():
                continue
            fields = entry.split("#")
            entry_dict = {}
            for field in fields:
                if "*" in field:
                    key, value = field.split("*", 1)
                    entry_dict[key] = value
            if entry_dict:
                parsed.append(entry_dict)

        return parsed

    def _fetch_km(self, ec_number: str, organism: Optional[str] = None) -> List[Dict]:
        """Fetch Km values from BRENDA API."""
        client = self._get_client()
        params = [
            self.email, self._get_password_hash(),
            f"ecNumber*{ec_number}",
            f"organism*{organism}" if organism else "organism*",
            "kmValue*", "kmValueMaximum*",
            "substrate*", "commentary*", "literature*"
        ]
        result = client.service.getKmValue(*params)
        return self._parse_brenda_result(result)

    def _fetch_kcat(self, ec_number: str, organism: Optional[str] = None) -> List[Dict]:
        """Fetch kcat values from BRENDA API."""
        client = self._get_client()
        params = [
            self.email, self._get_password_hash(),
            f"ecNumber*{ec_number}",
            f"organism*{organism}" if organism else "organism*",
            "turnoverNumber*", "turnoverNumberMaximum*",
            "substrate*", "commentary*", "literature*"
        ]
        result = client.service.getTurnoverNumber(*params)
        return self._parse_brenda_result(result)

    def _fetch_ki(self, ec_number: str, organism: Optional[str] = None) -> List[Dict]:
        """Fetch Ki values from BRENDA API."""
        client = self._get_client()
        params = [
            self.email, self._get_password_hash(),
            f"ecNumber*{ec_number}",
            f"organism*{organism}" if organism else "organism*",
            "kiValue*", "kiValueMaximum*",
            "inhibitor*", "commentary*", "literature*"
        ]
        result = client.service.getKiValue(*params)
        return self._parse_brenda_result(result)

    def load_enzyme(
        self,
        ec_number: str,
        organism: Optional[str] = None,
        use_cache: bool = True,
        use_api: bool = False,
    ) -> EnzymeKinetics:
        """
        Load kinetic data for an enzyme.

        Args:
            ec_number: EC number (e.g., "1.1.1.1")
            organism: Optional organism filter
            use_cache: Whether to use cached data
            use_api: Whether to fetch from API if not cached

        Returns:
            EnzymeKinetics object with all available parameters
        """
        cache_key = f"{ec_number}_{organism or 'all'}"
        cache_file = self.cache_dir / f"{cache_key.replace('.', '_')}.json"

        # Try loading from cache
        if use_cache and cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            return self._dict_to_enzyme_kinetics(data)

        # Fetch from API
        if use_api and self.email and self.password:
            km_data = self._fetch_km(ec_number, organism)
            kcat_data = self._fetch_kcat(ec_number, organism)
            ki_data = self._fetch_ki(ec_number, organism)

            enzyme = self._build_enzyme_kinetics(
                ec_number, organism, km_data, kcat_data, ki_data
            )

            # Save to cache
            if use_cache:
                with open(cache_file, "w") as f:
                    json.dump(self._enzyme_kinetics_to_dict(enzyme), f, indent=2)

            return enzyme

        # Return empty if no data available
        return EnzymeKinetics(ec_number=ec_number, organism=organism)

    def _build_enzyme_kinetics(
        self,
        ec_number: str,
        organism: Optional[str],
        km_data: List[Dict],
        kcat_data: List[Dict],
        ki_data: List[Dict],
    ) -> EnzymeKinetics:
        """Build EnzymeKinetics from parsed data."""
        enzyme = EnzymeKinetics(ec_number=ec_number, organism=organism)

        # Parse Km values
        for entry in km_data:
            try:
                value = float(entry.get("kmValue", 0))
                if value > 0:
                    param = KineticParameter(
                        ec_number=ec_number,
                        organism=entry.get("organism", organism or ""),
                        value=value,
                        value_max=float(entry.get("kmValueMaximum", value)) if entry.get("kmValueMaximum") else None,
                        substrate=entry.get("substrate", ""),
                        commentary=entry.get("commentary", ""),
                        literature=entry.get("literature", ""),
                    )
                    enzyme.km_values.append(param)
            except (ValueError, TypeError):
                continue

        # Parse kcat values
        for entry in kcat_data:
            try:
                value = float(entry.get("turnoverNumber", 0))
                if value > 0:
                    param = KineticParameter(
                        ec_number=ec_number,
                        organism=entry.get("organism", organism or ""),
                        value=value,
                        value_max=float(entry.get("turnoverNumberMaximum", value)) if entry.get("turnoverNumberMaximum") else None,
                        substrate=entry.get("substrate", ""),
                        commentary=entry.get("commentary", ""),
                        literature=entry.get("literature", ""),
                    )
                    enzyme.kcat_values.append(param)
            except (ValueError, TypeError):
                continue

        # Parse Ki values
        for entry in ki_data:
            try:
                value = float(entry.get("kiValue", 0))
                if value > 0:
                    param = KineticParameter(
                        ec_number=ec_number,
                        organism=entry.get("organism", organism or ""),
                        value=value,
                        value_max=float(entry.get("kiValueMaximum", value)) if entry.get("kiValueMaximum") else None,
                        inhibitor=entry.get("inhibitor", ""),
                        commentary=entry.get("commentary", ""),
                        literature=entry.get("literature", ""),
                    )
                    enzyme.ki_values.append(param)
            except (ValueError, TypeError):
                continue

        return enzyme

    def _enzyme_kinetics_to_dict(self, enzyme: EnzymeKinetics) -> Dict:
        """Convert EnzymeKinetics to dictionary for JSON serialization."""
        return {
            "ec_number": enzyme.ec_number,
            "organism": enzyme.organism,
            "km_values": [
                {
                    "ec_number": p.ec_number,
                    "organism": p.organism,
                    "value": p.value,
                    "value_max": p.value_max,
                    "substrate": p.substrate,
                    "commentary": p.commentary,
                    "literature": p.literature,
                }
                for p in enzyme.km_values
            ],
            "kcat_values": [
                {
                    "ec_number": p.ec_number,
                    "organism": p.organism,
                    "value": p.value,
                    "value_max": p.value_max,
                    "substrate": p.substrate,
                    "commentary": p.commentary,
                    "literature": p.literature,
                }
                for p in enzyme.kcat_values
            ],
            "ki_values": [
                {
                    "ec_number": p.ec_number,
                    "organism": p.organism,
                    "value": p.value,
                    "value_max": p.value_max,
                    "inhibitor": p.inhibitor,
                    "commentary": p.commentary,
                    "literature": p.literature,
                }
                for p in enzyme.ki_values
            ],
        }

    def _dict_to_enzyme_kinetics(self, data: Dict) -> EnzymeKinetics:
        """Convert dictionary back to EnzymeKinetics."""
        enzyme = EnzymeKinetics(
            ec_number=data["ec_number"],
            organism=data.get("organism"),
        )

        for km in data.get("km_values", []):
            enzyme.km_values.append(KineticParameter(
                ec_number=km["ec_number"],
                organism=km["organism"],
                value=km["value"],
                value_max=km.get("value_max"),
                substrate=km.get("substrate"),
                commentary=km.get("commentary", ""),
                literature=km.get("literature", ""),
            ))

        for kcat in data.get("kcat_values", []):
            enzyme.kcat_values.append(KineticParameter(
                ec_number=kcat["ec_number"],
                organism=kcat["organism"],
                value=kcat["value"],
                value_max=kcat.get("value_max"),
                substrate=kcat.get("substrate"),
                commentary=kcat.get("commentary", ""),
                literature=kcat.get("literature", ""),
            ))

        for ki in data.get("ki_values", []):
            enzyme.ki_values.append(KineticParameter(
                ec_number=ki["ec_number"],
                organism=ki["organism"],
                value=ki["value"],
                value_max=ki.get("value_max"),
                inhibitor=ki.get("inhibitor"),
                commentary=ki.get("commentary", ""),
                literature=ki.get("literature", ""),
            ))

        return enzyme

    def get_energy_parameters(
        self,
        enzyme: EnzymeKinetics,
        temperature: float = T_STANDARD,
    ) -> Dict[str, float]:
        """
        Convert kinetic parameters to Gibbs energy estimates.

        Returns:
            Dictionary with energy estimates in kJ/mol:
            - dg_binding: Substrate binding energy (from Km)
            - dg_activation: Activation energy (from kcat)
            - dg_inhibitor: Inhibitor binding energy (from Ki)
        """
        energies = {}

        km = enzyme.get_best_km()
        if km is not None:
            energies["dg_binding"] = km_to_binding_dg(km, temperature)

        kcat = enzyme.get_best_kcat()
        if kcat is not None:
            energies["dg_activation"] = kcat_to_activation_dg(kcat, temperature)

        ki = enzyme.get_best_ki()
        if ki is not None:
            energies["dg_inhibitor"] = ki_to_binding_dg(ki, temperature)

        return energies


def load_brenda_json(file_path: str) -> List[EnzymeKinetics]:
    """
    Load enzyme data from a BRENDA JSON export file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of EnzymeKinetics objects
    """
    with open(file_path) as f:
        data = json.load(f)

    loader = BRENDALoader()
    enzymes = []

    for entry in data:
        if isinstance(entry, dict) and "ec_number" in entry:
            enzyme = loader._dict_to_enzyme_kinetics(entry)
            enzymes.append(enzyme)

    return enzymes


def create_kinetic_parameter_dataset(
    enzymes: List[EnzymeKinetics],
    temperature: float = T_STANDARD,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create tensors of kinetic parameters from enzyme data.

    Returns:
        Tuple of (binding_energies, activation_energies, inhibitor_energies)
        Each tensor has shape (n_samples,) with valid values.
    """
    binding = []
    activation = []
    inhibitor = []

    for enzyme in enzymes:
        km = enzyme.get_best_km()
        if km is not None and km > 0:
            binding.append(km_to_binding_dg(km, temperature))

        kcat = enzyme.get_best_kcat()
        if kcat is not None and kcat > 0:
            activation.append(kcat_to_activation_dg(kcat, temperature))

        ki = enzyme.get_best_ki()
        if ki is not None and ki > 0:
            inhibitor.append(ki_to_binding_dg(ki, temperature))

    return (
        torch.tensor(binding, dtype=torch.float32) if binding else torch.tensor([]),
        torch.tensor(activation, dtype=torch.float32) if activation else torch.tensor([]),
        torch.tensor(inhibitor, dtype=torch.float32) if inhibitor else torch.tensor([]),
    )
