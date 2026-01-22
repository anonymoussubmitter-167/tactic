#!/usr/bin/env python
"""
Parse SLAC laccase kinetic data from raw plate reader files.

Data format:
- UTF-16 encoded text files
- Kinetic assay: absorbance measured over time
- Columns 1-10: different substrate (ABTS) concentrations
- Multiple replicates per timepoint
- Temperature recorded at each timepoint
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re


@dataclass
class KineticTrace:
    """Single kinetic trace (one substrate concentration)."""
    time: np.ndarray  # seconds
    absorbance: np.ndarray  # AU
    substrate_conc: float  # mM (will need calibration)
    temperature: float  # Celsius
    replicate: int


@dataclass
class SLACSample:
    """All kinetic data from one temperature."""
    temperature: float
    traces: List[KineticTrace]
    substrate_concentrations: List[float]


def parse_slac_raw_file(filepath: Path) -> SLACSample:
    """
    Parse a single SLAC raw data file.

    Returns kinetic traces for each substrate concentration.
    """
    # Read with UTF-16 encoding
    try:
        with open(filepath, 'r', encoding='utf-16') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='utf-16-le') as f:
            content = f.read()

    # Clean up the content - remove null bytes and normalize whitespace
    content = content.replace('\x00', '')
    lines = content.split('\n')

    # Extract temperature from filename
    temp_match = re.search(r'(\d+)\s*degrees?\s*C', filepath.name, re.IGNORECASE)
    if temp_match:
        temperature = float(temp_match.group(1))
    else:
        temperature = 25.0  # default

    # Parse data lines
    # Format: TIME    TEMP    col1    col2    ...    col10
    time_points = []
    data_blocks = []
    current_block = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_block:
                data_blocks.append(current_block)
                current_block = []
            continue

        # Check if this is a data line (starts with time like 00:00:00)
        time_match = re.match(r'^(\d{2}):(\d{2}):(\d{2})', line)
        if time_match:
            hours, mins, secs = map(int, time_match.groups())
            time_sec = hours * 3600 + mins * 60 + secs

            # Parse the rest of the line for absorbance values
            parts = line.split('\t')

            # Find numeric values (absorbance readings)
            values = []
            for part in parts[2:]:  # Skip time and temperature columns
                try:
                    val = float(part.strip())
                    values.append(val)
                except (ValueError, AttributeError):
                    pass

            if values:
                time_points.append(time_sec)
                current_block.append(values[:10])  # First 10 columns are sample data

        elif current_block and line.startswith('\t'):
            # Continuation line (replicate data)
            parts = line.split('\t')
            values = []
            for part in parts:
                try:
                    val = float(part.strip())
                    values.append(val)
                except (ValueError, AttributeError):
                    pass
            if values:
                current_block.append(values[:10])

    if current_block:
        data_blocks.append(current_block)

    # Organize into traces
    # Each block contains multiple replicates for one timepoint
    # We'll average replicates for now

    n_columns = 10  # 10 different substrate concentrations
    traces = []

    # Typical ABTS concentrations used (need to verify from calibration)
    # These are example values - actual values should come from calibration files
    substrate_concs = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # mM

    for col_idx in range(n_columns):
        col_data = []
        col_times = []

        for block_idx, block in enumerate(data_blocks):
            if block_idx < len(time_points):
                # Get all replicate values for this column
                col_values = [row[col_idx] for row in block if len(row) > col_idx]
                if col_values:
                    # Average replicates (excluding obvious outliers)
                    col_values = np.array(col_values)
                    # Simple outlier removal: within 2 std of mean
                    mean_val = np.mean(col_values)
                    std_val = np.std(col_values)
                    if std_val > 0:
                        mask = np.abs(col_values - mean_val) < 2 * std_val
                        col_values = col_values[mask]

                    col_data.append(np.mean(col_values))
                    col_times.append(time_points[block_idx])

        if col_data:
            trace = KineticTrace(
                time=np.array(col_times),
                absorbance=np.array(col_data),
                substrate_conc=substrate_concs[col_idx] if col_idx < len(substrate_concs) else 0.0,
                temperature=temperature,
                replicate=0,
            )
            traces.append(trace)

    return SLACSample(
        temperature=temperature,
        traces=traces,
        substrate_concentrations=substrate_concs[:len(traces)],
    )


def load_all_slac_data(data_dir: Path) -> Dict[float, SLACSample]:
    """Load all SLAC data files, organized by temperature."""
    raw_dir = data_dir / "enzymeml" / "slac" / "data" / "raw_data"

    samples = {}
    for txt_file in sorted(raw_dir.glob("*.txt")):
        print(f"Parsing: {txt_file.name}")
        try:
            sample = parse_slac_raw_file(txt_file)
            samples[sample.temperature] = sample
            print(f"  Temperature: {sample.temperature}°C")
            print(f"  Traces: {len(sample.traces)}")
            if sample.traces:
                print(f"  Time points: {len(sample.traces[0].time)}")
                print(f"  Time range: {sample.traces[0].time[0]:.0f} - {sample.traces[0].time[-1]:.0f} s")
        except Exception as e:
            print(f"  ERROR: {e}")

    return samples


def visualize_slac_data(samples: Dict[float, SLACSample]):
    """Create a simple text visualization of the data."""
    print("\n" + "="*70)
    print("SLAC DATA SUMMARY")
    print("="*70)

    for temp, sample in sorted(samples.items()):
        print(f"\nTemperature: {temp}°C")
        print("-" * 40)

        for trace in sample.traces[:3]:  # Show first 3 traces
            print(f"  [S] = {trace.substrate_conc:.3f} mM:")
            print(f"    Time: {trace.time[0]:.0f} - {trace.time[-1]:.0f} s ({len(trace.time)} points)")
            print(f"    Abs:  {trace.absorbance[0]:.4f} - {trace.absorbance[-1]:.4f}")

            # Estimate initial rate (slope of first few points)
            if len(trace.time) > 2:
                dt = trace.time[1] - trace.time[0]
                dA = trace.absorbance[1] - trace.absorbance[0]
                v0 = dA / dt * 60  # AU/min
                print(f"    v0 ≈ {v0:.4f} AU/min")


def main():
    data_dir = Path(__file__).parent.parent / "data" / "real"

    print("="*70)
    print("SLAC Data Parser")
    print("="*70)

    samples = load_all_slac_data(data_dir)

    if samples:
        visualize_slac_data(samples)

        # Save parsed data
        import pickle
        output_path = data_dir / "slac_parsed.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"\nSaved parsed data to: {output_path}")

    return samples


if __name__ == "__main__":
    main()
