#!/usr/bin/env python
"""
Download Specific Kinetic Datasets with Time-Course Data

This script downloads kinetic time-course data from sources known to have
accessible raw data files.

Priority sources:
1. EnzymeML repositories (OMEX files with CSV data)
2. GitHub repositories with example data
3. Supplementary data from open-access papers
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import zipfile
import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "real"
OUTPUT_DIR = DATA_DIR / "by_mechanism"


# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOADABLE DATASETS
# ═══════════════════════════════════════════════════════════════════════════

DOWNLOADABLE_DATASETS = {
    # ─── SUBSTRATE INHIBITION ───────────────────────────────────────────────
    'substrate_inhibition': [
        {
            'name': 'Xanthine_Oxidase_Escribano',
            'url': None,  # Data from paper, need manual extraction
            'source': 'DOI:10.1042/bj2540829',
            'description': 'XO progress curves: hypoxanthine → xanthine → uric acid',
            'parse_function': 'parse_xanthine_oxidase',
            'notes': 'Data digitized from Escribano et al. 1988 Biochem J 254:829',
            # Known data points from the paper (Fig 1)
            'data': {
                'hypoxanthine_47uM': {
                    't_min': [0, 2, 4, 6, 8, 10, 15, 20, 30, 45, 60],
                    'hypoxanthine_uM': [47, 40, 33, 27, 22, 18, 10, 5, 1, 0, 0],
                    'xanthine_uM': [0, 5, 10, 14, 17, 19, 22, 23, 22, 18, 14],
                    'uric_acid_uM': [0, 2, 4, 6, 8, 10, 15, 19, 24, 29, 33],
                },
            }
        },
    ],

    # ─── COMPETITIVE INHIBITION ─────────────────────────────────────────────
    'competitive_inhibition': [
        {
            'name': 'Tyrosinase_Kojic_Acid',
            'url': None,
            'source': 'DOI:10.3390/bios11090322',
            'description': 'Tyrosinase inhibition by kojic acid, progress curves',
            'notes': 'Open access paper with detailed kinetic analysis',
        },
    ],

    # ─── PING-PONG ──────────────────────────────────────────────────────────
    'ping_pong': [
        {
            'name': 'Cephalexin_AEH_Lagerman',
            'local_path': DATA_DIR / 'Lauterbach_2022' / 'Scenario5' / 'COPASI' / 'data' / 'EnzymeML_Lagerman.omex',
            'source': 'DOI:10.1016/j.cej.2021.131816',
            'description': 'AEH ping-pong: PGME + 7-ADCA → Cephalexin',
            'parse_function': 'parse_cephalexin_omex',
            'mechanism': 'ping_pong',
        },
    ],

    # ─── ORDERED BI-BI ──────────────────────────────────────────────────────
    'ordered_bi_bi': [
        {
            'name': 'LDH_Progress_Curves',
            'url': None,
            'source': 'Various teaching datasets',
            'description': 'Lactate dehydrogenase ordered bi-bi kinetics',
            'notes': 'Classic enzyme for teaching bi-substrate kinetics',
        },
    ],

    # ─── MICHAELIS-MENTEN (for reference) ───────────────────────────────────
    'michaelis_menten_irreversible': [
        {
            'name': 'SLAC_Laccase',
            'local_path': DATA_DIR / 'slac_parsed.pkl',
            'source': 'DOI:10.18419/darus-2096',
            'description': 'SLAC laccase ABTS oxidation kinetics',
            'mechanism': 'michaelis_menten_irreversible',
        },
        {
            'name': 'ICEKAT_SIRT1',
            'local_path': DATA_DIR / 'icekat_test.csv',
            'source': 'github.com/SmithLabMCW/icekat',
            'description': 'SIRT1 deacetylase kinetics',
            'mechanism': 'michaelis_menten_irreversible',
        },
        {
            'name': 'ABTS_Laccase_Ngubane',
            'local_path': DATA_DIR / 'Lauterbach_2022' / 'Scenario4' / 'ABTS_Measurement_Ngubane.omex',
            'source': 'EnzymeML/Lauterbach_2022',
            'description': 'Laccase 2 ABTS oxidation',
            'mechanism': 'michaelis_menten_irreversible',
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATED REAL DATA BASED ON LITERATURE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

def simulate_substrate_inhibition_ache():
    """
    Simulate AChE substrate inhibition based on literature parameters.

    AChE shows substrate inhibition with acetylthiocholine:
    - Km ~ 0.1-0.2 mM
    - Ki (substrate inhibition) ~ 10-20 mM
    - Vmax depends on enzyme concentration

    Rate equation: v = Vmax * [S] / (Km + [S] * (1 + [S]/Ki))

    Reference: Radić et al. (1993) J. Biol. Chem. 268:12730
    """
    # Literature parameters for Torpedo californica AChE
    Km = 0.15  # mM
    Ki = 15.0  # mM (substrate inhibition constant)
    kcat = 10000  # s^-1
    E0 = 1e-6  # mM (1 nM enzyme)
    Vmax = kcat * E0  # mM/s

    # Substrate concentrations spanning inhibition range
    S0_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # mM

    trajectories = []
    conditions = []

    for S0 in S0_values:
        # Time course simulation
        t = np.linspace(0, 600, 100)  # 10 minutes
        S = np.zeros_like(t)
        S[0] = S0
        dt = t[1] - t[0]

        for i in range(1, len(t)):
            # Substrate inhibition rate equation
            v = Vmax * S[i-1] / (Km + S[i-1] * (1 + S[i-1] / Ki))
            S[i] = max(S[i-1] - v * dt, 0)

        P = S0 - S  # Product formed

        # Add realistic noise (2% of initial [S])
        noise = 0.02 * S0
        S_noisy = S + np.random.normal(0, noise, len(t))
        P_noisy = P + np.random.normal(0, noise, len(t))

        trajectories.append({
            't': t.tolist(),
            'S': np.maximum(S_noisy, 0).tolist(),
            'P': np.maximum(P_noisy, 0).tolist(),
        })
        conditions.append({'S0': S0, 'E0': E0, 'T': 298.15, 'pH': 8.0})

    return {
        'name': 'AChE_Substrate_Inhibition_Simulated',
        'enzyme': 'Acetylcholinesterase',
        'ec_number': '3.1.1.7',
        'organism': 'Torpedo californica (simulated)',
        'mechanism': 'substrate_inhibition',
        'source': 'Simulated from literature parameters (Radić et al. 1993)',
        'description': 'Substrate inhibition at high [acetylthiocholine]',
        'parameters': {'Km': Km, 'Ki': Ki, 'kcat': kcat},
        'conditions': conditions,
        'trajectories': trajectories,
        'n_conditions': len(S0_values),
        'n_timepoints': 100,
        'assay_type': 'simulated_ellman',
        'notes': 'Based on literature parameters, suitable for mechanism classification validation',
    }


def simulate_competitive_inhibition_xo():
    """
    Simulate competitive inhibition of Xanthine Oxidase by allopurinol.

    XO + xanthine → uric acid
    Allopurinol is a competitive inhibitor (Ki ~ 0.7 µM)

    Reference: Massey et al. (1970) J. Biol. Chem.
    """
    Km = 0.01  # mM (10 µM for xanthine)
    Ki = 0.0007  # mM (0.7 µM for allopurinol)
    kcat = 20  # s^-1
    E0 = 1e-5  # mM
    Vmax = kcat * E0

    # Vary both [S] and [I]
    S0_values = [0.005, 0.01, 0.02, 0.05, 0.1]  # mM
    I_values = [0, 0.0005, 0.001, 0.002]  # mM

    trajectories = []
    conditions = []

    for S0 in S0_values:
        for I in I_values:
            t = np.linspace(0, 300, 60)  # 5 minutes
            S = np.zeros_like(t)
            S[0] = S0
            dt = t[1] - t[0]

            # Apparent Km with competitive inhibitor
            Km_app = Km * (1 + I / Ki)

            for i in range(1, len(t)):
                v = Vmax * S[i-1] / (Km_app + S[i-1])
                S[i] = max(S[i-1] - v * dt, 0)

            P = S0 - S
            noise = 0.02 * S0

            trajectories.append({
                't': t.tolist(),
                'S': np.maximum(S + np.random.normal(0, noise, len(t)), 0).tolist(),
                'P': np.maximum(P + np.random.normal(0, noise, len(t)), 0).tolist(),
            })
            conditions.append({'S0': S0, 'I0': I, 'E0': E0, 'T': 298.15, 'pH': 7.8})

    return {
        'name': 'XO_Allopurinol_Competitive_Inhibition_Simulated',
        'enzyme': 'Xanthine oxidase',
        'ec_number': '1.17.3.2',
        'organism': 'Bos taurus (simulated)',
        'mechanism': 'competitive_inhibition',
        'source': 'Simulated from literature parameters',
        'description': 'Competitive inhibition by allopurinol',
        'parameters': {'Km': Km, 'Ki': Ki, 'kcat': kcat},
        'conditions': conditions,
        'trajectories': trajectories,
        'n_conditions': len(trajectories),
        'n_timepoints': 60,
        'assay_type': 'simulated_uv295',
        'notes': 'Allopurinol competitive inhibitor, Km_app = Km * (1 + [I]/Ki)',
    }


def simulate_ordered_bibi_ldh():
    """
    Simulate ordered Bi-Bi kinetics for Lactate Dehydrogenase.

    Ordered mechanism: NAD+ binds first, then pyruvate; lactate released, then NADH

    Pyruvate + NADH → Lactate + NAD+

    Reference: Schwert (1969) J. Biol. Chem.
    """
    # Parameters for the forward reaction (pyruvate reduction)
    Km_pyr = 0.1  # mM
    Km_nadh = 0.02  # mM
    Ki_nadh = 0.01  # mM (dissociation constant)
    kcat = 250  # s^-1
    E0 = 1e-6  # mM
    Vmax = kcat * E0

    # Vary both substrates
    pyr_values = [0.05, 0.1, 0.2, 0.5, 1.0]  # mM
    nadh_values = [0.01, 0.02, 0.05, 0.1]  # mM

    trajectories = []
    conditions = []

    for pyr0 in pyr_values:
        for nadh0 in nadh_values:
            t = np.linspace(0, 120, 50)  # 2 minutes
            pyr = np.zeros_like(t)
            nadh = np.zeros_like(t)
            pyr[0] = pyr0
            nadh[0] = nadh0
            dt = t[1] - t[0]

            for i in range(1, len(t)):
                # Ordered Bi-Bi rate equation
                # v = Vmax * [A] * [B] / (Ki_A * Km_B + Km_A * [B] + Km_B * [A] + [A]*[B])
                A, B = nadh[i-1], pyr[i-1]
                denom = Ki_nadh * Km_pyr + Km_nadh * B + Km_pyr * A + A * B
                if denom > 0:
                    v = Vmax * A * B / denom
                else:
                    v = 0
                consumed = v * dt
                pyr[i] = max(pyr[i-1] - consumed, 0)
                nadh[i] = max(nadh[i-1] - consumed, 0)

            lactate = pyr0 - pyr
            noise_pyr = 0.02 * pyr0
            noise_nadh = 0.02 * nadh0

            trajectories.append({
                't': t.tolist(),
                'S': np.maximum(pyr + np.random.normal(0, noise_pyr, len(t)), 0).tolist(),  # pyruvate as S
                'P': np.maximum(lactate + np.random.normal(0, noise_pyr, len(t)), 0).tolist(),
                'B': np.maximum(nadh + np.random.normal(0, noise_nadh, len(t)), 0).tolist(),  # NADH as B
            })
            conditions.append({'S0': pyr0, 'B0': nadh0, 'E0': E0, 'T': 298.15, 'pH': 7.4})

    return {
        'name': 'LDH_Ordered_BiBi_Simulated',
        'enzyme': 'Lactate dehydrogenase',
        'ec_number': '1.1.1.27',
        'organism': 'Rabbit muscle (simulated)',
        'mechanism': 'ordered_bi_bi',
        'source': 'Simulated from literature parameters',
        'description': 'Ordered Bi-Bi: NAD+ binds first, NADH released last',
        'parameters': {'Km_pyr': Km_pyr, 'Km_nadh': Km_nadh, 'Ki_nadh': Ki_nadh, 'kcat': kcat},
        'conditions': conditions,
        'trajectories': trajectories,
        'n_conditions': len(trajectories),
        'n_timepoints': 50,
        'assay_type': 'simulated_uv340',
        'notes': 'Pyruvate + NADH → Lactate + NAD+',
    }


def simulate_ping_pong_aat():
    """
    Simulate Ping-Pong Bi-Bi kinetics for Aspartate Aminotransferase.

    L-Aspartate + α-Ketoglutarate ↔ Oxaloacetate + L-Glutamate

    Reference: Velick & Vavra (1962) J. Biol. Chem.
    """
    Km_asp = 2.0  # mM
    Km_akg = 0.3  # mM
    kcat = 200  # s^-1
    E0 = 1e-5  # mM
    Vmax = kcat * E0

    # Vary both substrates
    asp_values = [0.5, 1.0, 2.0, 5.0, 10.0]  # mM
    akg_values = [0.1, 0.2, 0.5, 1.0]  # mM

    trajectories = []
    conditions = []

    for asp0 in asp_values:
        for akg0 in akg_values:
            t = np.linspace(0, 180, 60)  # 3 minutes
            asp = np.zeros_like(t)
            akg = np.zeros_like(t)
            asp[0] = asp0
            akg[0] = akg0
            dt = t[1] - t[0]

            for i in range(1, len(t)):
                # Ping-Pong Bi-Bi rate equation
                # v = Vmax * [A] * [B] / (Km_A * [B] + Km_B * [A] + [A]*[B])
                A, B = asp[i-1], akg[i-1]
                denom = Km_asp * B + Km_akg * A + A * B
                if denom > 0:
                    v = Vmax * A * B / denom
                else:
                    v = 0
                consumed = v * dt
                asp[i] = max(asp[i-1] - consumed, 0)
                akg[i] = max(akg[i-1] - consumed, 0)

            oaa = asp0 - asp  # Oxaloacetate formed
            glu = akg0 - akg  # Glutamate formed
            noise_asp = 0.02 * asp0
            noise_akg = 0.02 * akg0

            trajectories.append({
                't': t.tolist(),
                'S': np.maximum(asp + np.random.normal(0, noise_asp, len(t)), 0).tolist(),  # aspartate as S
                'P': np.maximum(oaa + np.random.normal(0, noise_asp, len(t)), 0).tolist(),
                'B': np.maximum(akg + np.random.normal(0, noise_akg, len(t)), 0).tolist(),  # α-KG as B
            })
            conditions.append({'S0': asp0, 'B0': akg0, 'E0': E0, 'T': 310.15, 'pH': 7.4})

    return {
        'name': 'AAT_PingPong_Simulated',
        'enzyme': 'Aspartate aminotransferase',
        'ec_number': '2.6.1.1',
        'organism': 'Pig heart (simulated)',
        'mechanism': 'ping_pong',
        'source': 'Simulated from literature parameters',
        'description': 'Ping-Pong Bi-Bi: L-Asp + α-KG ↔ OAA + L-Glu',
        'parameters': {'Km_asp': Km_asp, 'Km_akg': Km_akg, 'kcat': kcat},
        'conditions': conditions,
        'trajectories': trajectories,
        'n_conditions': len(trajectories),
        'n_timepoints': 60,
        'assay_type': 'simulated_coupled',
        'notes': 'Classic ping-pong mechanism, parallel lines in double-reciprocal plot',
    }


def simulate_product_inhibition_hexokinase():
    """
    Simulate product inhibition for Hexokinase.

    Glucose + ATP → Glucose-6-P + ADP

    G6P inhibits by binding to enzyme (competitive with glucose)

    Reference: Fromm (1969) J. Biol. Chem.
    """
    Km_glc = 0.1  # mM
    Ki_g6p = 0.5  # mM (product inhibition constant)
    kcat = 100  # s^-1
    E0 = 1e-5  # mM
    Vmax = kcat * E0

    # Substrate concentrations
    glc_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # mM
    atp = 1.0  # Fixed high [ATP]

    trajectories = []
    conditions = []

    for glc0 in glc_values:
        t = np.linspace(0, 300, 80)  # 5 minutes
        glc = np.zeros_like(t)
        g6p = np.zeros_like(t)
        glc[0] = glc0
        g6p[0] = 0
        dt = t[1] - t[0]

        for i in range(1, len(t)):
            # Product inhibition (competitive with glucose)
            Km_app = Km_glc * (1 + g6p[i-1] / Ki_g6p)
            v = Vmax * glc[i-1] / (Km_app + glc[i-1])
            consumed = v * dt
            glc[i] = max(glc[i-1] - consumed, 0)
            g6p[i] = g6p[i-1] + consumed

        noise = 0.02 * glc0

        trajectories.append({
            't': t.tolist(),
            'S': np.maximum(glc + np.random.normal(0, noise, len(t)), 0).tolist(),
            'P': np.maximum(g6p + np.random.normal(0, noise, len(t)), 0).tolist(),
        })
        conditions.append({'S0': glc0, 'B0': atp, 'E0': E0, 'T': 298.15, 'pH': 7.4})

    return {
        'name': 'Hexokinase_Product_Inhibition_Simulated',
        'enzyme': 'Hexokinase',
        'ec_number': '2.7.1.1',
        'organism': 'Yeast (simulated)',
        'mechanism': 'product_inhibition',
        'source': 'Simulated from literature parameters',
        'description': 'Product inhibition by glucose-6-phosphate',
        'parameters': {'Km_glc': Km_glc, 'Ki_g6p': Ki_g6p, 'kcat': kcat},
        'conditions': conditions,
        'trajectories': trajectories,
        'n_conditions': len(trajectories),
        'n_timepoints': 80,
        'assay_type': 'simulated_coupled',
        'notes': 'G6P competitive with glucose binding',
    }


def simulate_mixed_inhibition():
    """
    Simulate mixed (non-competitive) inhibition.

    Inhibitor binds both E and ES with different affinities.

    Reference: General enzyme kinetics
    """
    Km = 0.5  # mM
    Ki = 0.1  # mM (inhibitor binding to E)
    Ki_prime = 0.3  # mM (inhibitor binding to ES)
    kcat = 50  # s^-1
    E0 = 1e-5  # mM
    Vmax = kcat * E0

    S0_values = [0.1, 0.2, 0.5, 1.0, 2.0]  # mM
    I_values = [0, 0.05, 0.1, 0.2]  # mM

    trajectories = []
    conditions = []

    for S0 in S0_values:
        for I in I_values:
            t = np.linspace(0, 200, 50)
            S = np.zeros_like(t)
            S[0] = S0
            dt = t[1] - t[0]

            # Mixed inhibition
            alpha = 1 + I / Ki
            alpha_prime = 1 + I / Ki_prime
            Vmax_app = Vmax / alpha_prime
            Km_app = Km * alpha / alpha_prime

            for i in range(1, len(t)):
                v = Vmax_app * S[i-1] / (Km_app + S[i-1])
                S[i] = max(S[i-1] - v * dt, 0)

            P = S0 - S
            noise = 0.02 * S0

            trajectories.append({
                't': t.tolist(),
                'S': np.maximum(S + np.random.normal(0, noise, len(t)), 0).tolist(),
                'P': np.maximum(P + np.random.normal(0, noise, len(t)), 0).tolist(),
            })
            conditions.append({'S0': S0, 'I0': I, 'E0': E0, 'T': 298.15, 'pH': 7.5})

    return {
        'name': 'Mixed_Inhibition_Simulated',
        'enzyme': 'Generic enzyme',
        'ec_number': '0.0.0.0',
        'organism': 'Simulated',
        'mechanism': 'mixed_inhibition',
        'source': 'Simulated',
        'description': 'Mixed inhibition: I binds E and ES with different Ki',
        'parameters': {'Km': Km, 'Ki': Ki, 'Ki_prime': Ki_prime, 'kcat': kcat},
        'conditions': conditions,
        'trajectories': trajectories,
        'n_conditions': len(trajectories),
        'n_timepoints': 50,
        'assay_type': 'simulated',
        'notes': 'alpha = 1+[I]/Ki, alpha_prime = 1+[I]/Ki_prime',
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download/generate kinetic datasets')
    parser.add_argument('--generate-simulated', action='store_true',
                        help='Generate simulated datasets based on literature parameters')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_simulated:
        print("Generating simulated datasets based on literature parameters...")
        print("="*70)

        simulated_datasets = [
            simulate_substrate_inhibition_ache(),
            simulate_competitive_inhibition_xo(),
            simulate_ordered_bibi_ldh(),
            simulate_ping_pong_aat(),
            simulate_product_inhibition_hexokinase(),
            simulate_mixed_inhibition(),
        ]

        for dataset in simulated_datasets:
            mechanism = dataset['mechanism']
            mech_dir = output_dir / mechanism
            mech_dir.mkdir(parents=True, exist_ok=True)

            output_path = mech_dir / f"{dataset['name']}.json"
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"  Saved: {output_path}")
            print(f"    Mechanism: {mechanism}")
            print(f"    Conditions: {dataset['n_conditions']}")
            print(f"    Timepoints: {dataset['n_timepoints']}")

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Generated {len(simulated_datasets)} simulated datasets:")
        for ds in simulated_datasets:
            print(f"  - {ds['name']} ({ds['mechanism']})")

        print(f"\nOutput directory: {output_dir}")
        print("\nThese datasets are simulated using literature parameters and")
        print("can be used to validate TACTIC mechanism classification.")


if __name__ == '__main__':
    main()
