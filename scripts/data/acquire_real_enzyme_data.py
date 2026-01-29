#!/usr/bin/env python
"""
Acquire Real Enzyme Kinetic Data by Mechanism Type

This script downloads and organizes real enzyme kinetic time-course data
from multiple sources for validating TACTIC mechanism classification.

Sources:
1. SABIO-RK Database (API queries)
2. EnzymeML repositories (GitHub)
3. Published literature (supplementary data)
4. BRENDA database (parameters + references)

Target mechanisms:
- michaelis_menten_irreversible (MM-I)
- michaelis_menten_reversible (MM-R)
- competitive_inhibition (CI)
- uncompetitive_inhibition (UI)
- mixed_inhibition (MI)
- substrate_inhibition (SI)
- ordered_bi_bi (OBB)
- random_bi_bi (RBB)
- ping_pong (PP)
- product_inhibition (PI)
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import xml.etree.ElementTree as ET
import zipfile
import io

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "real" / "by_mechanism"


@dataclass
class EnzymeDataset:
    """Container for enzyme kinetic time-course data."""
    name: str
    enzyme: str
    ec_number: str
    organism: str
    mechanism: str  # Our TACTIC mechanism label
    source: str  # Paper DOI, database ID, or URL
    description: str

    # Experimental conditions
    conditions: List[Dict] = field(default_factory=list)  # [{S0, E0, I0, T, pH, ...}, ...]

    # Time-course data: list of trajectories
    # Each trajectory: {t: array, S: array, P: array, ...}
    trajectories: List[Dict] = field(default_factory=list)

    # Metadata
    n_conditions: int = 0
    n_timepoints: int = 0
    assay_type: str = ""  # spectrophotometric, fluorescence, etc.
    notes: str = ""

    def to_dict(self):
        d = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        for i, traj in enumerate(d['trajectories']):
            for key, val in traj.items():
                if isinstance(val, np.ndarray):
                    d['trajectories'][i][key] = val.tolist()
        return d


# ═══════════════════════════════════════════════════════════════════════════
# KNOWN ENZYME DATA SOURCES BY MECHANISM
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_ENZYMES = {
    'substrate_inhibition': [
        {
            'name': 'Acetylcholinesterase (AChE)',
            'enzyme': 'Acetylcholinesterase',
            'ec': '3.1.1.7',
            'organism': 'Torpedo californica / Electrophorus electricus',
            'source': 'DOI:10.1038/sj.emboj.7601175',
            'description': 'Classic substrate inhibition at high acetylthiocholine concentrations',
            'assay': 'Ellman assay (DTNB, 412nm)',
            'notes': 'Bell-shaped velocity curve, peripheral anionic site binding',
        },
        {
            'name': 'Xanthine Oxidase',
            'enzyme': 'Xanthine oxidase',
            'ec': '1.17.3.2',
            'organism': 'Bos taurus (bovine milk)',
            'source': 'DOI:10.1152/advan.00107.2012',
            'description': 'Substrate inhibition by hypoxanthine/xanthine >50 µM',
            'assay': 'UV absorbance at 295nm (uric acid)',
            'notes': 'v = Vmax*[S]/(Km + a*[S] + b*[S]^2/Ki)',
        },
        {
            'name': 'Tyrosinase',
            'enzyme': 'Tyrosinase (polyphenol oxidase)',
            'ec': '1.14.18.1',
            'organism': 'Agaricus bisporus (mushroom)',
            'source': 'DOI:10.1038/srep34993',
            'description': 'Substrate inhibition with L-DOPA and other phenolic substrates',
            'assay': 'UV-Vis at 475nm (dopachrome)',
            'notes': 'Complex inhibition pattern with multiple binding sites',
        },
    ],

    'competitive_inhibition': [
        {
            'name': 'Tyrosinase + Kojic acid',
            'enzyme': 'Tyrosinase',
            'ec': '1.14.18.1',
            'organism': 'Agaricus bisporus',
            'source': 'DOI:10.3390/bios11090322',
            'description': 'Kojic acid competitive inhibition, progress curve analysis',
            'assay': 'UV-Vis at 475nm',
            'notes': 't1/2 vs [I] plots at varying [S]',
        },
        {
            'name': 'AChE + Galantamine',
            'enzyme': 'Acetylcholinesterase',
            'ec': '3.1.1.7',
            'organism': 'Electrophorus electricus',
            'source': 'DOI:10.3390/ijms23084764',
            'description': 'Time-dependent competitive inhibition by galantamine',
            'assay': 'Ellman assay',
            'notes': 'Anti-Alzheimer drug, slow-binding inhibitor',
        },
        {
            'name': 'Xanthine Oxidase + Allopurinol',
            'enzyme': 'Xanthine oxidase',
            'ec': '1.17.3.2',
            'organism': 'Bos taurus',
            'source': 'PMID:6274312',
            'description': 'Alloxanthine (allopurinol metabolite) competitive inhibition',
            'assay': 'UV absorbance',
            'notes': 'Ki = 35 nM, slow-binding',
        },
    ],

    'product_inhibition': [
        {
            'name': 'Hexokinase',
            'enzyme': 'Hexokinase',
            'ec': '2.7.1.1',
            'organism': 'Saccharomyces cerevisiae / tumor cells',
            'source': 'JBC 245:6292-6299 (1970)',
            'description': 'Product inhibition by glucose-6-phosphate and ADP',
            'assay': 'Coupled assay (G6PDH/NADPH)',
            'notes': 'Mixed inhibition by ADP when ATP is varied',
        },
        {
            'name': 'Fumarase',
            'enzyme': 'Fumarate hydratase',
            'ec': '4.2.1.2',
            'organism': 'Pig heart',
            'source': 'DOI:10.1038/srep02658',
            'description': 'Product inhibition in reversible fumarate-malate conversion',
            'assay': 'UV at 250nm (fumarate)',
            'notes': 'Full time course analysis with product inhibition',
        },
    ],

    'ordered_bi_bi': [
        {
            'name': 'Lactate Dehydrogenase (LDH)',
            'enzyme': 'L-lactate dehydrogenase',
            'ec': '1.1.1.27',
            'organism': 'Bos taurus (heart) / rabbit muscle',
            'source': 'DOI:10.1111/febs.13972',
            'description': 'Ordered Bi-Bi: NAD+ binds first, NADH released last',
            'assay': 'UV at 340nm (NADH)',
            'notes': 'Pyruvate + NADH ↔ Lactate + NAD+',
        },
        {
            'name': 'Alcohol Dehydrogenase (ADH)',
            'enzyme': 'Alcohol dehydrogenase',
            'ec': '1.1.1.1',
            'organism': 'Equus caballus (horse liver)',
            'source': 'ResearchGate:18793007',
            'description': 'Ordered Bi-Bi mechanism',
            'assay': 'UV at 340nm (NADH)',
            'notes': 'Ethanol + NAD+ ↔ Acetaldehyde + NADH',
        },
    ],

    'random_bi_bi': [
        {
            'name': 'Creatine Kinase',
            'enzyme': 'Creatine kinase',
            'ec': '2.7.3.2',
            'organism': 'Oryctolagus cuniculus (rabbit muscle)',
            'source': 'DOI:10.1080/10409230590918577',
            'description': 'Random Bi-Bi at pH 8, ordered at lower pH',
            'assay': 'Coupled assay or 31P NMR',
            'notes': 'ATP + Creatine ↔ ADP + Phosphocreatine',
        },
        {
            'name': 'Yeast Alcohol Dehydrogenase',
            'enzyme': 'Alcohol dehydrogenase I',
            'ec': '1.1.1.1',
            'organism': 'Saccharomyces cerevisiae',
            'source': 'PMID:15112059',
            'description': 'Random Bi-Bi with 2-propanol/NAD+',
            'assay': 'UV at 340nm',
            'notes': 'Dead-end complexes formed',
        },
    ],

    'ping_pong': [
        {
            'name': 'Aspartate Aminotransferase (AAT)',
            'enzyme': 'Aspartate aminotransferase',
            'ec': '2.6.1.1',
            'organism': 'Escherichia coli / pig heart',
            'source': 'DOI:10.1016/j.bbapap.2014.01.002',
            'description': 'Classic ping-pong: L-Asp + α-KG ↔ OAA + L-Glu',
            'assay': 'Coupled assay (MDH)',
            'notes': 'PLP-dependent, parallel lines in double-reciprocal plot',
        },
        {
            'name': 'Branched-Chain Aminotransferase (BCAT)',
            'enzyme': 'Branched-chain amino acid aminotransferase',
            'ec': '2.6.1.42',
            'organism': 'Escherichia coli',
            'source': 'PMID:24206068',
            'description': 'Ping-pong Bi-Bi with BCAAs',
            'assay': 'Coupled assay',
            'notes': 'Ile/Leu/Val + α-KG ↔ keto acid + Glu',
        },
        {
            'name': 'Alanine Transaminase (ALT)',
            'enzyme': 'Alanine transaminase',
            'ec': '2.6.1.2',
            'organism': 'Various',
            'source': 'PMID:1662617',
            'description': 'Ping-pong: L-Ala + α-KG ↔ Pyruvate + L-Glu',
            'assay': 'Coupled assay (LDH)',
            'notes': 'Clinical biomarker',
        },
        {
            'name': 'Cephalexin AEH (existing)',
            'enzyme': 'α-Amino ester hydrolase',
            'ec': '3.1.1.43',
            'organism': 'Xanthomonas campestris',
            'source': 'DOI:10.1016/j.cej.2021.131816',
            'description': 'PGME + 7-ADCA → Cephalexin (Ping-Pong)',
            'assay': 'HPLC',
            'notes': 'Already in Lauterbach_2022 Scenario 5',
        },
    ],

    'mixed_inhibition': [
        {
            'name': 'Hexokinase + ADP',
            'enzyme': 'Hexokinase',
            'ec': '2.7.1.1',
            'organism': 'Saccharomyces cerevisiae',
            'source': 'JBC 245:6292-6299 (1970)',
            'description': 'Mixed inhibition by ADP (product)',
            'assay': 'Coupled assay',
            'notes': 'Affects both Km and Vmax',
        },
    ],

    'uncompetitive_inhibition': [
        {
            'name': 'LDH + Pyruvate (high)',
            'enzyme': 'Lactate dehydrogenase',
            'ec': '1.1.1.27',
            'organism': 'Duck epsilon-crystallin',
            'source': 'PMID:1989512',
            'description': 'Uncompetitive substrate inhibition by pyruvate',
            'assay': 'UV at 340nm',
            'notes': 'Ki = 6.7 mM',
        },
    ],

    'michaelis_menten_reversible': [
        {
            'name': 'Triose Phosphate Isomerase (TIM)',
            'enzyme': 'Triosephosphate isomerase',
            'ec': '5.3.1.1',
            'organism': 'Various',
            'source': 'DOI:10.1073/pnas.0608876104',
            'description': 'DHAP ↔ G3P, near diffusion limit',
            'assay': 'Coupled assay or NMR',
            'notes': 'Keq ~ 0.05 (favors DHAP)',
        },
        {
            'name': 'Fumarase',
            'enzyme': 'Fumarate hydratase',
            'ec': '4.2.1.2',
            'organism': 'Pig heart',
            'source': 'DOI:10.1038/srep02658',
            'description': 'Fumarate ↔ Malate, reversible',
            'assay': 'UV at 250nm',
            'notes': 'Diffusion-limited enzyme',
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# SABIO-RK API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

SABIO_BASE_URL = "https://sabiork.h-its.org/sabioRestWebServices"

def query_sabio_rk(query: str, format: str = 'sbml', max_results: int = 100) -> Optional[str]:
    """Query SABIO-RK database."""
    url = f"{SABIO_BASE_URL}/searchKineticLaws/{format}"
    params = {'q': query}

    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            return response.text
        else:
            print(f"  SABIO-RK query failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  SABIO-RK request error: {e}")
        return None


def search_sabio_by_enzyme(enzyme_name: str, ec_number: str = None) -> List[Dict]:
    """Search SABIO-RK for kinetic data by enzyme name or EC number."""
    results = []

    # Try EC number first (more specific)
    if ec_number:
        query = f"ECNumber:{ec_number}"
        data = query_sabio_rk(query)
        if data:
            results.extend(parse_sabio_sbml(data))

    # Also try enzyme name
    query = f"EnzymeName:*{enzyme_name}*"
    data = query_sabio_rk(query)
    if data:
        results.extend(parse_sabio_sbml(data))

    return results


def parse_sabio_sbml(sbml_text: str) -> List[Dict]:
    """Parse SABIO-RK SBML response."""
    results = []

    try:
        root = ET.fromstring(sbml_text)
        # Parse SBML structure for kinetic laws
        # This is a simplified parser - full implementation would extract
        # reaction details, kinetic parameters, and any time-course data

        for model in root.iter('{http://www.sbml.org/sbml/level2/version4}model'):
            model_id = model.get('id', 'unknown')
            results.append({'model_id': model_id, 'raw': sbml_text[:1000]})
    except ET.ParseError as e:
        print(f"  SBML parse error: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENZYMEML / OMEX FILE PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_omex_file(omex_path: Path) -> Optional[EnzymeDataset]:
    """Parse an EnzymeML OMEX file to extract kinetic data."""
    if not omex_path.exists():
        return None

    try:
        with zipfile.ZipFile(omex_path, 'r') as zf:
            # Find experiment.xml
            xml_files = [f for f in zf.namelist() if f.endswith('.xml') and 'experiment' in f.lower()]
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]

            experiment_data = None
            if xml_files:
                with zf.open(xml_files[0]) as f:
                    experiment_data = f.read().decode('utf-8')

            # Parse CSV data files
            trajectories = []
            for csv_file in sorted(csv_files):
                with zf.open(csv_file) as f:
                    df = pd.read_csv(f)
                    traj = {col: df[col].values for col in df.columns}
                    trajectories.append(traj)

            return {
                'xml': experiment_data,
                'trajectories': trajectories,
                'csv_files': csv_files,
            }
    except Exception as e:
        print(f"  Error parsing OMEX {omex_path}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def download_github_file(url: str, output_path: Path) -> bool:
    """Download a file from GitHub."""
    try:
        # Convert GitHub URL to raw URL if needed
        if 'github.com' in url and '/blob/' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"  Download error: {e}")
    return False


def download_supplementary_data(doi: str, output_dir: Path) -> List[Path]:
    """Attempt to download supplementary data from a paper DOI."""
    # This would require publisher-specific APIs
    # For now, return empty - manual download may be needed
    print(f"  Note: Manual download may be required for DOI: {doi}")
    return []


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_substrate_inhibition(
    S0_range: Tuple[float, float] = (0.01, 10.0),
    n_conditions: int = 10,
    n_timepoints: int = 50,
    Vmax: float = 1.0,
    Km: float = 0.5,
    Ki: float = 5.0,
    noise: float = 0.02,
) -> EnzymeDataset:
    """Generate synthetic substrate inhibition data for testing."""
    S0_values = np.logspace(np.log10(S0_range[0]), np.log10(S0_range[1]), n_conditions)

    trajectories = []
    conditions = []

    for S0 in S0_values:
        # Substrate inhibition rate: v = Vmax*S / (Km + S + S^2/Ki)
        t = np.linspace(0, 100, n_timepoints)

        # Simple numerical integration
        S = np.zeros(n_timepoints)
        S[0] = S0
        dt = t[1] - t[0]

        for i in range(1, n_timepoints):
            v = Vmax * S[i-1] / (Km + S[i-1] + S[i-1]**2 / Ki)
            S[i] = max(S[i-1] - v * dt, 0)

        P = S0 - S

        # Add noise
        S_noisy = S + np.random.normal(0, noise * S0, n_timepoints)
        P_noisy = P + np.random.normal(0, noise * S0, n_timepoints)

        trajectories.append({
            't': t,
            'S': np.maximum(S_noisy, 0),
            'P': np.maximum(P_noisy, 0),
        })
        conditions.append({'S0': S0, 'E0': 0.001})

    return EnzymeDataset(
        name='Synthetic Substrate Inhibition',
        enzyme='Test Enzyme',
        ec_number='0.0.0.0',
        organism='Synthetic',
        mechanism='substrate_inhibition',
        source='Generated',
        description=f'Synthetic data: Vmax={Vmax}, Km={Km}, Ki={Ki}',
        conditions=conditions,
        trajectories=trajectories,
        n_conditions=n_conditions,
        n_timepoints=n_timepoints,
        assay_type='synthetic',
        notes=f'Noise level: {noise*100}%',
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ACQUISITION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def acquire_data_for_mechanism(mechanism: str, output_dir: Path) -> List[EnzymeDataset]:
    """Acquire all available data for a specific mechanism."""
    datasets = []

    if mechanism not in KNOWN_ENZYMES:
        print(f"Unknown mechanism: {mechanism}")
        return datasets

    print(f"\n{'='*60}")
    print(f"Acquiring data for: {mechanism}")
    print(f"{'='*60}")

    for enzyme_info in KNOWN_ENZYMES[mechanism]:
        print(f"\n  {enzyme_info['name']}...")

        # Try SABIO-RK
        print(f"    Querying SABIO-RK for EC {enzyme_info['ec']}...")
        sabio_results = search_sabio_by_enzyme(
            enzyme_info['enzyme'],
            enzyme_info['ec']
        )
        if sabio_results:
            print(f"    Found {len(sabio_results)} SABIO-RK entries")

        # Create dataset entry (even without time-course data)
        dataset = EnzymeDataset(
            name=enzyme_info['name'],
            enzyme=enzyme_info['enzyme'],
            ec_number=enzyme_info['ec'],
            organism=enzyme_info['organism'],
            mechanism=mechanism,
            source=enzyme_info['source'],
            description=enzyme_info['description'],
            assay_type=enzyme_info.get('assay', ''),
            notes=enzyme_info.get('notes', ''),
        )
        datasets.append(dataset)

    return datasets


def generate_mechanism_summary(output_dir: Path):
    """Generate a summary of all acquired data by mechanism."""
    summary = {
        'generated': datetime.now().isoformat(),
        'mechanisms': {},
    }

    for mechanism in KNOWN_ENZYMES:
        mech_dir = output_dir / mechanism
        datasets = []

        if mech_dir.exists():
            for json_file in mech_dir.glob('*.json'):
                with open(json_file) as f:
                    datasets.append(json.load(f))

        summary['mechanisms'][mechanism] = {
            'n_datasets': len(KNOWN_ENZYMES[mechanism]),
            'known_enzymes': [e['name'] for e in KNOWN_ENZYMES[mechanism]],
            'acquired_datasets': len(datasets),
        }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Acquire real enzyme kinetic data')
    parser.add_argument('--mechanism', type=str, default=None,
                        help='Specific mechanism to acquire (default: all)')
    parser.add_argument('--list', action='store_true',
                        help='List known enzymes by mechanism')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic test data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("\n" + "="*70)
        print("KNOWN ENZYMES BY MECHANISM TYPE")
        print("="*70)

        for mechanism, enzymes in KNOWN_ENZYMES.items():
            print(f"\n{mechanism} ({len(enzymes)} enzymes):")
            for e in enzymes:
                print(f"  - {e['name']} (EC {e['ec']})")
                print(f"    Source: {e['source']}")
        return

    if args.synthetic:
        print("\nGenerating synthetic test data...")

        # Generate synthetic data for testing
        for mechanism in ['substrate_inhibition']:
            dataset = generate_synthetic_substrate_inhibition()

            mech_dir = output_dir / mechanism
            mech_dir.mkdir(parents=True, exist_ok=True)

            output_path = mech_dir / f"synthetic_{mechanism}.json"
            with open(output_path, 'w') as f:
                json.dump(dataset.to_dict(), f, indent=2)
            print(f"  Saved: {output_path}")
        return

    # Acquire real data
    mechanisms = [args.mechanism] if args.mechanism else list(KNOWN_ENZYMES.keys())

    all_datasets = {}
    for mechanism in mechanisms:
        datasets = acquire_data_for_mechanism(mechanism, output_dir)
        all_datasets[mechanism] = datasets

        # Save datasets
        mech_dir = output_dir / mechanism
        mech_dir.mkdir(parents=True, exist_ok=True)

        for i, dataset in enumerate(datasets):
            safe_name = dataset.name.replace(' ', '_').replace('/', '_').replace('+', '_')
            output_path = mech_dir / f"{safe_name}.json"
            with open(output_path, 'w') as f:
                json.dump(dataset.to_dict(), f, indent=2)
            print(f"    Saved: {output_path}")

    # Generate summary
    summary = generate_mechanism_summary(output_dir)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for mech, info in summary['mechanisms'].items():
        print(f"  {mech}: {info['n_datasets']} known enzymes")

    print(f"\nOutput directory: {output_dir}")
    print("\nNOTE: Most datasets require manual download of time-course data.")
    print("See the JSON files for source DOIs and download instructions.")


if __name__ == '__main__':
    main()
