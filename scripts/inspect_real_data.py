#!/usr/bin/env python
"""
Inspect real enzyme kinetics datasets for TACTIC evaluation.
"""

import numpy as np
from pathlib import Path
import json


def inspect_enzymeml_slac(data_dir: Path):
    """
    Parse and inspect EnzymeML SLAC laccase data.
    """
    print("\n" + "="*70)
    print("EnzymeML SLAC-ABTS Dataset (Laccase)")
    print("="*70)
    print("Known mechanism: Michaelis-Menten irreversible")
    print("Source: https://github.com/EnzymeML/slac_modeling")
    print()

    omex_dir = data_dir / "enzymeml" / "slac" / "data" / "enzymeml"

    if not omex_dir.exists():
        print(f"ERROR: Directory not found: {omex_dir}")
        return None

    omex_files = sorted(omex_dir.glob("*.omex"))
    print(f"Found {len(omex_files)} .omex files:")
    for f in omex_files:
        print(f"  - {f.name}")

    # Try to parse with pyenzyme
    try:
        from pyenzyme import EnzymeMLDocument
    except ImportError:
        print("\nWARNING: pyenzyme not installed. Install with: pip install pyenzyme")
        return inspect_enzymeml_slac_raw(data_dir)

    all_data = []

    for omex_file in omex_files:
        print(f"\n--- Parsing: {omex_file.name} ---")

        try:
            doc = EnzymeMLDocument.fromFile(str(omex_file))

            # Extract temperature from filename
            temp_str = omex_file.stem.split("_")[-1].replace("C", "")
            temperature = float(temp_str)

            print(f"  Temperature: {temperature}°C")

            # Proteins/enzymes
            if hasattr(doc, 'proteins') and doc.proteins:
                for pid, protein in doc.proteins.items():
                    print(f"  Enzyme: {protein.name} (ID: {pid})")

            # Reactants
            if hasattr(doc, 'small_molecules') and doc.small_molecules:
                print(f"  Small molecules: {len(doc.small_molecules)}")
                for sid, mol in doc.small_molecules.items():
                    print(f"    - {mol.name} (ID: {sid})")

            # Measurements
            if hasattr(doc, 'measurements') and doc.measurements:
                print(f"  Measurements: {len(doc.measurements)}")

                for mid, measurement in doc.measurements.items():
                    print(f"\n  Measurement {mid}:")

                    # Get species data
                    if hasattr(measurement, 'species_data') and measurement.species_data:
                        for species_id, species_data in measurement.species_data.items():
                            print(f"    Species: {species_id}")

                            # Check for replicates
                            if hasattr(species_data, 'replicates') and species_data.replicates:
                                for rep in species_data.replicates:
                                    if hasattr(rep, 'time_unit'):
                                        print(f"      Time unit: {rep.time_unit}")
                                    if hasattr(rep, 'data_unit'):
                                        print(f"      Data unit: {rep.data_unit}")
                                    if hasattr(rep, 'time') and rep.time:
                                        t = np.array(rep.time)
                                        print(f"      Time points: {len(t)} ({t.min():.2f} - {t.max():.2f})")
                                    if hasattr(rep, 'data') and rep.data:
                                        d = np.array(rep.data)
                                        print(f"      Data range: {d.min():.6f} - {d.max():.6f}")

                                        # Store for analysis
                                        all_data.append({
                                            'temperature': temperature,
                                            'species_id': species_id,
                                            'time': t.tolist() if hasattr(rep, 'time') else [],
                                            'data': d.tolist() if hasattr(rep, 'data') else [],
                                        })

        except Exception as e:
            print(f"  ERROR parsing: {e}")
            import traceback
            traceback.print_exc()

    return all_data


def inspect_enzymeml_slac_raw(data_dir: Path):
    """
    Inspect raw SLAC data files if pyenzyme fails.
    """
    print("\n--- Inspecting raw data files ---")

    raw_dir = data_dir / "enzymeml" / "slac" / "data" / "raw_data"
    if not raw_dir.exists():
        print(f"Raw data directory not found: {raw_dir}")
        return None

    for txt_file in sorted(raw_dir.glob("*.txt")):
        print(f"\n{txt_file.name}:")
        with open(txt_file) as f:
            lines = f.readlines()
            print(f"  Lines: {len(lines)}")
            print(f"  First 5 lines:")
            for line in lines[:5]:
                print(f"    {line.strip()[:80]}")

    # Also check calibration files
    cal_dir = data_dir / "enzymeml" / "slac" / "data" / "calibrations"
    if cal_dir.exists():
        print("\n--- Calibration files ---")
        for json_file in sorted(cal_dir.glob("*.json")):
            print(f"\n{json_file.name}:")
            with open(json_file) as f:
                data = json.load(f)
                print(f"  Keys: {list(data.keys())[:10]}")

    return None


def summarize_for_tactic(data_dir: Path):
    """
    Summarize what data is available for TACTIC evaluation.
    """
    print("\n" + "="*70)
    print("SUMMARY FOR TACTIC EVALUATION")
    print("="*70)

    datasets = []

    # Check SLAC
    slac_dir = data_dir / "enzymeml" / "slac"
    if slac_dir.exists():
        omex_files = list((slac_dir / "data" / "enzymeml").glob("*.omex"))
        datasets.append({
            'name': 'SLAC Laccase (EnzymeML)',
            'source': 'GitHub/EnzymeML',
            'mechanism': 'michaelis_menten_irreversible',
            'files': len(omex_files),
            'conditions': 'Temperature (25-45°C)',
            'status': 'AVAILABLE',
        })

    # Print summary
    print("\nAvailable datasets:")
    for ds in datasets:
        print(f"\n  {ds['name']}:")
        print(f"    Status: {ds['status']}")
        print(f"    Source: {ds['source']}")
        print(f"    Known mechanism: {ds['mechanism']}")
        print(f"    Conditions: {ds['conditions']}")
        print(f"    Files: {ds['files']}")

    print("\n" + "-"*70)
    print("Evaluation plan:")
    print("-"*70)
    print("""
    1. SLAC Laccase (MM irreversible):
       - Model should classify as 'michaelis_menten_irreversible'
       - Test: Does our model generalize to real data?
       - Challenge: Only temperature variation (not [S] or [I])

    2. For bi-substrate and inhibition mechanisms:
       - Need to find additional datasets from BRENDA/SABIO-RK
       - Or use SLAC data to test basic generalization first
    """)

    return datasets


def main():
    data_dir = Path(__file__).parent.parent / "data" / "real"

    print("="*70)
    print("TACTIC Real Data Inspector")
    print("="*70)
    print(f"Data directory: {data_dir}")

    # Inspect SLAC
    slac_data = inspect_enzymeml_slac(data_dir)

    # Summary
    summarize_for_tactic(data_dir)

    return slac_data


if __name__ == "__main__":
    main()
