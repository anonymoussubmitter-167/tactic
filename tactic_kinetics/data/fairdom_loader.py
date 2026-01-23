"""
FAIRDOMHub data loader for enzyme kinetics experiments.

FAIRDOMHub (https://fairdomhub.org/) is a repository for FAIR research data,
including enzyme kinetics experiments with progress curves and kinetic parameters.

This module provides tools for loading:
1. Excel files exported from FAIRDOMHub
2. JCAMP-DX spectroscopy files
3. SBtab formatted data

The data typically includes:
- Time-course measurements (progress curves)
- Experimental conditions (temperature, pH, concentrations)
- Enzyme and substrate information
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset


@dataclass
class KineticsExperiment:
    """
    A single enzyme kinetics experiment from FAIRDOMHub.

    Contains time-course data with experimental conditions.
    """
    experiment_id: str
    enzyme_name: str
    substrate_name: str

    # Time-course data
    times: np.ndarray  # Time points (seconds)
    concentrations: np.ndarray  # Measured concentrations (mM)
    species_name: str = "product"  # What was measured

    # Experimental conditions
    temperature: Optional[float] = None  # K
    ph: Optional[float] = None
    enzyme_conc: Optional[float] = None  # mM
    substrate_conc: Optional[float] = None  # mM initial
    inhibitor_name: Optional[str] = None
    inhibitor_conc: Optional[float] = None  # mM

    # Metadata
    source: str = ""
    notes: str = ""

    @property
    def n_timepoints(self) -> int:
        return len(self.times)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors."""
        return (
            torch.tensor(self.times, dtype=torch.float32),
            torch.tensor(self.concentrations, dtype=torch.float32),
        )


@dataclass
class KineticsDataset:
    """Collection of kinetics experiments from FAIRDOMHub."""
    experiments: List[KineticsExperiment] = field(default_factory=list)
    source_file: Optional[str] = None

    def __len__(self) -> int:
        return len(self.experiments)

    def __getitem__(self, idx: int) -> KineticsExperiment:
        return self.experiments[idx]

    def filter_by_enzyme(self, enzyme_name: str) -> "KineticsDataset":
        """Filter experiments by enzyme name."""
        filtered = [e for e in self.experiments if enzyme_name.lower() in e.enzyme_name.lower()]
        return KineticsDataset(experiments=filtered, source_file=self.source_file)

    def filter_by_substrate(self, substrate_name: str) -> "KineticsDataset":
        """Filter experiments by substrate name."""
        filtered = [e for e in self.experiments if substrate_name.lower() in e.substrate_name.lower()]
        return KineticsDataset(experiments=filtered, source_file=self.source_file)


class FAIRDOMLoader:
    """
    Loader for FAIRDOMHub enzyme kinetics data.

    Supports loading from:
    - Excel files (.xlsx, .xls)
    - CSV files
    - SBtab files
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize FAIRDOMHub loader.

        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/fairdom")

    def load_excel(
        self,
        file_path: str,
        time_column: str = "Time",
        value_column: str = "Product",
        sheet_name: Optional[str] = None,
        metadata_sheet: Optional[str] = None,
    ) -> KineticsDataset:
        """
        Load kinetics data from an Excel file.

        Supports common FAIRDOMHub Excel formats with:
        - Time-course data in one sheet
        - Metadata in another sheet (optional)

        Args:
            file_path: Path to Excel file
            time_column: Name of time column
            value_column: Name of value column (product, substrate, etc.)
            sheet_name: Name of data sheet (None = first sheet)
            metadata_sheet: Name of metadata sheet

        Returns:
            KineticsDataset with loaded experiments
        """
        file_path = Path(file_path)
        if not file_path.exists():
            file_path = self.data_dir / file_path

        # Read Excel file
        try:
            if sheet_name is None:
                df = pd.read_excel(file_path, sheet_name=0)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

        # Load metadata if available
        metadata = {}
        if metadata_sheet:
            try:
                meta_df = pd.read_excel(file_path, sheet_name=metadata_sheet)
                for _, row in meta_df.iterrows():
                    if "Parameter" in meta_df.columns and "Value" in meta_df.columns:
                        metadata[row["Parameter"]] = row["Value"]
            except Exception:
                pass

        # Parse time-course data
        experiments = self._parse_timecourse_df(
            df, time_column, value_column, metadata, file_path.name
        )

        return KineticsDataset(experiments=experiments, source_file=str(file_path))

    def _parse_timecourse_df(
        self,
        df: pd.DataFrame,
        time_column: str,
        value_column: str,
        metadata: Dict,
        source_name: str,
    ) -> List[KineticsExperiment]:
        """Parse a DataFrame containing time-course data."""
        experiments = []

        # Find time column
        time_col = None
        for col in df.columns:
            if time_column.lower() in col.lower():
                time_col = col
                break

        if time_col is None:
            # Try common alternatives
            for alt in ["time", "t", "Time (s)", "Time (min)"]:
                for col in df.columns:
                    if alt.lower() in col.lower():
                        time_col = col
                        break
                if time_col:
                    break

        if time_col is None:
            raise ValueError(f"Could not find time column '{time_column}'")

        # Find value columns
        value_cols = []
        for col in df.columns:
            if col != time_col:
                if value_column.lower() in col.lower() or col != time_col:
                    value_cols.append(col)

        # Create experiments for each value column
        times = df[time_col].values.astype(float)

        for i, val_col in enumerate(value_cols):
            values = df[val_col].values

            # Skip non-numeric columns
            try:
                values = values.astype(float)
            except (ValueError, TypeError):
                continue

            # Skip all-NaN columns
            if np.isnan(values).all():
                continue

            exp = KineticsExperiment(
                experiment_id=f"{source_name}_{i}",
                enzyme_name=metadata.get("Enzyme", "Unknown"),
                substrate_name=metadata.get("Substrate", "Unknown"),
                times=times,
                concentrations=values,
                species_name=val_col,
                temperature=metadata.get("Temperature", 298.15),
                ph=metadata.get("pH"),
                enzyme_conc=metadata.get("Enzyme_concentration"),
                substrate_conc=metadata.get("Substrate_concentration"),
                source=source_name,
            )
            experiments.append(exp)

        return experiments

    def load_csv(
        self,
        file_path: str,
        time_column: str = "Time",
        value_column: str = "Product",
        delimiter: str = ",",
    ) -> KineticsDataset:
        """
        Load kinetics data from a CSV file.

        Args:
            file_path: Path to CSV file
            time_column: Name of time column
            value_column: Name of value column
            delimiter: CSV delimiter

        Returns:
            KineticsDataset with loaded experiments
        """
        file_path = Path(file_path)
        if not file_path.exists():
            file_path = self.data_dir / file_path

        df = pd.read_csv(file_path, delimiter=delimiter)
        experiments = self._parse_timecourse_df(
            df, time_column, value_column, {}, file_path.name
        )

        return KineticsDataset(experiments=experiments, source_file=str(file_path))

    def load_sbtab(self, file_path: str) -> KineticsDataset:
        """
        Load kinetics data from an SBtab file.

        SBtab is a standardized format for Systems Biology data tables.
        See: https://www.sbtab.net/

        Args:
            file_path: Path to SBtab file

        Returns:
            KineticsDataset with loaded experiments
        """
        file_path = Path(file_path)
        if not file_path.exists():
            file_path = self.data_dir / file_path

        # SBtab files start with !!SBtab header
        with open(file_path) as f:
            lines = f.readlines()

        # Parse header
        header = {}
        data_lines = []
        columns = None

        for line in lines:
            line = line.strip()
            if line.startswith("!!SBtab"):
                # Parse header attributes
                parts = line.split("\t")
                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        header[key.strip("!")] = value.strip("'\"")
            elif line.startswith("!"):
                # Column names
                columns = [c.strip("!") for c in line.split("\t")]
            elif line and columns:
                # Data row
                data_lines.append(line.split("\t"))

        if not columns or not data_lines:
            raise ValueError("Could not parse SBtab file")

        # Create DataFrame
        df = pd.DataFrame(data_lines, columns=columns)

        # Parse as time-course
        experiments = self._parse_timecourse_df(
            df, "Time", "Concentration", header, file_path.name
        )

        return KineticsDataset(experiments=experiments, source_file=str(file_path))

    def load_directory(
        self,
        directory: Optional[str] = None,
        pattern: str = "*.xlsx",
    ) -> KineticsDataset:
        """
        Load all matching files from a directory.

        Args:
            directory: Directory to search (defaults to data_dir)
            pattern: Glob pattern for files

        Returns:
            Combined KineticsDataset
        """
        directory = Path(directory) if directory else self.data_dir
        all_experiments = []

        for file_path in directory.glob(pattern):
            try:
                if file_path.suffix.lower() in [".xlsx", ".xls"]:
                    dataset = self.load_excel(str(file_path))
                elif file_path.suffix.lower() == ".csv":
                    dataset = self.load_csv(str(file_path))
                else:
                    continue

                all_experiments.extend(dataset.experiments)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

        return KineticsDataset(experiments=all_experiments)


class FAIRDOMTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for FAIRDOMHub kinetics data.

    Converts KineticsDataset to format suitable for training.
    """

    def __init__(
        self,
        kinetics_dataset: KineticsDataset,
        max_timepoints: int = 50,
        normalize: bool = True,
    ):
        """
        Initialize PyTorch dataset.

        Args:
            kinetics_dataset: FAIRDOMHub kinetics dataset
            max_timepoints: Maximum number of time points (pad/truncate)
            normalize: Whether to normalize concentrations
        """
        self.experiments = kinetics_dataset.experiments
        self.max_timepoints = max_timepoints
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.experiments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        exp = self.experiments[idx]

        # Get time and concentration data
        times = exp.times
        values = exp.concentrations

        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not valid_mask.all():
            values = np.interp(times, times[valid_mask], values[valid_mask])

        # Normalize
        if self.normalize and values.max() > 0:
            values = values / values.max()

        # Pad/truncate to max_timepoints
        n_points = len(times)
        if n_points > self.max_timepoints:
            # Subsample
            indices = np.linspace(0, n_points - 1, self.max_timepoints, dtype=int)
            times = times[indices]
            values = values[indices]
            mask = np.ones(self.max_timepoints, dtype=bool)
        elif n_points < self.max_timepoints:
            # Pad
            pad_length = self.max_timepoints - n_points
            times = np.concatenate([times, np.zeros(pad_length)])
            values = np.concatenate([values, np.zeros(pad_length)])
            mask = np.concatenate([np.ones(n_points, dtype=bool), np.zeros(pad_length, dtype=bool)])
        else:
            mask = np.ones(self.max_timepoints, dtype=bool)

        # Build conditions tensor
        conditions = np.array([
            exp.temperature or 298.15,
            exp.ph or 7.0,
            exp.substrate_conc or 1.0,
            exp.enzyme_conc or 0.01,
        ])

        return {
            "times": torch.tensor(times, dtype=torch.float32),
            "values": torch.tensor(values, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "conditions": torch.tensor(conditions, dtype=torch.float32),
            "experiment_id": exp.experiment_id,
        }


def load_fairdom_excel(file_path: str, **kwargs) -> KineticsDataset:
    """Convenience function to load a FAIRDOMHub Excel file."""
    loader = FAIRDOMLoader()
    return loader.load_excel(file_path, **kwargs)


def create_torch_dataset(
    kinetics_dataset: KineticsDataset,
    **kwargs
) -> FAIRDOMTorchDataset:
    """Convenience function to create a PyTorch dataset."""
    return FAIRDOMTorchDataset(kinetics_dataset, **kwargs)
