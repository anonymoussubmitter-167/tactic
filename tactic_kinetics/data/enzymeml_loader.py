"""
EnzymeML data loading for TACTIC-Kinetics.

This module provides utilities for loading EnzymeML documents
(OMEX archives) and converting them to PyTorch datasets.
"""

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET


class EnzymeMLDocument:
    """
    Parser for EnzymeML documents (OMEX archives).

    EnzymeML documents contain:
    - Experimental conditions (temperature, pH, concentrations)
    - Time course measurements
    - Model parameters (if fitted)
    - Reaction definitions
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.measurements = []
        self.conditions = {}
        self.species = {}
        self.reactions = []
        self.parameters = {}

        self._parse()

    def _parse(self):
        """Parse the OMEX archive."""
        if self.path.suffix == ".omex":
            self._parse_omex()
        elif self.path.suffix == ".json":
            self._parse_json()
        else:
            raise ValueError(f"Unsupported file format: {self.path.suffix}")

    def _parse_omex(self):
        """Parse OMEX (zip) archive."""
        with zipfile.ZipFile(self.path, "r") as archive:
            # Find the main EnzymeML file
            files = archive.namelist()

            # Look for JSON or XML files
            for fname in files:
                if fname.endswith(".json"):
                    with archive.open(fname) as f:
                        data = json.load(f)
                        self._parse_enzymeml_json(data)
                        return
                elif fname.endswith(".xml"):
                    with archive.open(fname) as f:
                        self._parse_sbml(f.read())
                        return

    def _parse_json(self):
        """Parse standalone JSON file."""
        with open(self.path) as f:
            data = json.load(f)
            self._parse_enzymeml_json(data)

    def _parse_enzymeml_json(self, data: Dict):
        """Parse EnzymeML JSON format."""
        # Extract measurements
        if "measurements" in data:
            for meas in data["measurements"]:
                self.measurements.append({
                    "id": meas.get("id"),
                    "name": meas.get("name"),
                    "species_data": meas.get("species", []),
                })

        # Extract species
        if "species" in data:
            for sp in data["species"]:
                self.species[sp.get("id")] = {
                    "name": sp.get("name"),
                    "initial_concentration": sp.get("init_conc"),
                    "unit": sp.get("unit"),
                }

        # Extract reactions
        if "reactions" in data:
            self.reactions = data["reactions"]

        # Extract conditions
        if "conditions" in data:
            self.conditions = data["conditions"]

        # Extract fitted parameters
        if "parameters" in data:
            self.parameters = data["parameters"]

    def _parse_sbml(self, xml_content: bytes):
        """Parse SBML format."""
        root = ET.fromstring(xml_content)

        # SBML namespace handling
        ns = {"sbml": "http://www.sbml.org/sbml/level3/version1/core"}

        # Extract species
        for species in root.findall(".//sbml:species", ns):
            sp_id = species.get("id")
            self.species[sp_id] = {
                "name": species.get("name", sp_id),
                "initial_concentration": float(species.get("initialConcentration", 0)),
            }

        # Extract reactions
        for reaction in root.findall(".//sbml:reaction", ns):
            self.reactions.append({
                "id": reaction.get("id"),
                "name": reaction.get("name"),
            })

    def get_time_series(self, species_id: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract time series data for species.

        Returns:
            Dict mapping species ID to (times, values) arrays
        """
        result = {}

        for meas in self.measurements:
            for sp_data in meas.get("species_data", []):
                sp_id = sp_data.get("species_id") or sp_data.get("id")

                if species_id is not None and sp_id != species_id:
                    continue

                times = np.array(sp_data.get("time", []))
                values = np.array(sp_data.get("data", sp_data.get("values", [])))

                if len(times) > 0 and len(values) > 0:
                    result[sp_id] = (times, values)

        return result

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "path": str(self.path),
            "measurements": self.measurements,
            "species": self.species,
            "reactions": self.reactions,
            "conditions": self.conditions,
            "parameters": self.parameters,
        }


def load_enzymeml_omex(path: str) -> EnzymeMLDocument:
    """
    Load an EnzymeML OMEX file.

    Args:
        path: Path to the OMEX file

    Returns:
        Parsed EnzymeML document
    """
    return EnzymeMLDocument(path)


class EnzymeMLDataset(Dataset):
    """
    PyTorch Dataset for EnzymeML data.

    Loads progress curves from EnzymeML documents for training.
    """

    def __init__(
        self,
        documents: List[EnzymeMLDocument],
        max_obs: int = 50,
        observable_species: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Args:
            documents: List of parsed EnzymeML documents
            max_obs: Maximum number of observations per curve
            observable_species: Which species to use as observable (None = auto-detect)
            normalize: Whether to normalize values
        """
        self.documents = documents
        self.max_obs = max_obs
        self.observable_species = observable_species
        self.normalize = normalize

        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict]:
        """Prepare samples from documents."""
        samples = []

        for doc in self.documents:
            time_series = doc.get_time_series(self.observable_species)

            for sp_id, (times, values) in time_series.items():
                # Extract conditions
                conditions = self._extract_conditions(doc)

                sample = {
                    "times": times,
                    "values": values,
                    "conditions": conditions,
                    "species_id": sp_id,
                    "document_path": str(doc.path),
                }

                # Normalize if requested
                if self.normalize and len(values) > 0:
                    sample["values"] = values / (values.max() + 1e-8)

                samples.append(sample)

        return samples

    def _extract_conditions(self, doc: EnzymeMLDocument) -> np.ndarray:
        """Extract experimental conditions as array."""
        conditions = doc.conditions

        # Default conditions
        T = conditions.get("temperature", 298.15)
        if isinstance(T, dict):
            T = T.get("value", 298.15)

        pH = conditions.get("ph", 7.0)
        if isinstance(pH, dict):
            pH = pH.get("value", 7.0)

        # Try to get initial concentrations
        S0 = 1.0
        E0 = 0.01
        for sp_id, sp_info in doc.species.items():
            init_conc = sp_info.get("initial_concentration", 0)
            if "substrate" in sp_id.lower() or sp_id == "S":
                S0 = init_conc
            elif "enzyme" in sp_id.lower() or sp_id == "E":
                E0 = init_conc

        return np.array([T, pH, S0, E0], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.samples[idx]

        times = sample["times"]
        values = sample["values"]
        n_obs = len(times)

        # Pad or truncate to max_obs
        if n_obs >= self.max_obs:
            times = times[:self.max_obs]
            values = values[:self.max_obs]
            mask = np.ones(self.max_obs, dtype=bool)
        else:
            pad_len = self.max_obs - n_obs
            times = np.concatenate([times, np.zeros(pad_len)])
            values = np.concatenate([values, np.zeros(pad_len)])
            mask = np.concatenate([np.ones(n_obs, dtype=bool), np.zeros(pad_len, dtype=bool)])

        return {
            "times": torch.tensor(times, dtype=torch.float32),
            "values": torch.tensor(values, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "conditions": torch.tensor(sample["conditions"], dtype=torch.float32),
        }


def load_slac_dataset(data_dir: str = "data/enzymeml") -> EnzymeMLDataset:
    """
    Load the SLAC laccase dataset.

    Args:
        data_dir: Directory containing OMEX files

    Returns:
        EnzymeMLDataset
    """
    data_path = Path(data_dir)

    # Find all OMEX files
    omex_files = list(data_path.glob("**/*.omex"))

    if len(omex_files) == 0:
        raise FileNotFoundError(f"No OMEX files found in {data_dir}")

    # Load all documents
    documents = [EnzymeMLDocument(f) for f in omex_files]

    return EnzymeMLDataset(documents)
