#!/usr/bin/env python
"""
Experiment 8: Real Data Evaluation

Test TACTIC vs Classical AIC on real enzyme kinetic data from 3 sources:

1. SLAC Laccase (EnzymeML/DaRUS)
   - Enzyme: Small laccase (Streptomyces coelicolor), EC 1.10.3.2
   - Assay: ABTS oxidation, absorbance at 420nm
   - Data: 5 temperatures × 10 [S] = 50 traces, 11 timepoints each, 900s
   - Known mechanism: Michaelis-Menten irreversible
   - Source: DOI 10.18419/darus-2096

2. ICEKAT SIRT1 (SmithLabMCW/GitHub)
   - Enzyme: SIRT1 protein lysine deacetylase
   - Assay: Fluorescence/absorbance kinetic trace
   - Data: 9 substrate concentrations (0-500 µM), 76 timepoints, 600s
   - Known mechanism: Michaelis-Menten irreversible
   - Source: github.com/SmithLabMCW/icekat

3. ABTS Laccase (EnzymeML/Lauterbach_2022 Scenario 4)
   - Enzyme: Laccase 2 (Trametes pubescens), EC 1.10.3.2
   - Assay: ABTS oxidation, substrate concentration (µmol/L) over time
   - Data: 9 [S] (6.5-149 µM), 21 timepoints each, 1200s, 3 replicates
   - Known mechanism: Michaelis-Menten irreversible
   - Source: github.com/EnzymeML/Lauterbach_2022

This validates TACTIC generalization from synthetic training data to
real experimental measurements.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Needed for pickle deserialization of SLACSample
import importlib
parse_slac = importlib.import_module('parse_slac_data')
sys.modules['__main__'].SLACSample = parse_slac.SLACSample
sys.modules['__main__'].KineticTrace = parse_slac.KineticTrace

import json
import time
import numpy as np
import torch
import pickle
from datetime import datetime
from collections import defaultdict

from tactic_kinetics.models.multi_condition_classifier import (
    create_multi_task_model,
    create_multi_condition_model,
    create_basic_multi_condition_model,
)
from tactic_kinetics.training.multi_condition_generator import MultiConditionSample

MECHANISMS = [
    'michaelis_menten_irreversible',
    'michaelis_menten_reversible',
    'competitive_inhibition',
    'uncompetitive_inhibition',
    'mixed_inhibition',
    'substrate_inhibition',
    'ordered_bi_bi',
    'random_bi_bi',
    'ping_pong',
    'product_inhibition',
]


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADERS: Convert real data → MultiConditionSample + TACTIC tensors
# ═══════════════════════════════════════════════════════════════════════

def load_slac_data(base_dir: Path):
    """Load SLAC laccase parsed data."""
    from parse_slac_data import SLACSample, KineticTrace

    slac_path = base_dir / "data" / "real" / "slac_parsed.pkl"
    if not slac_path.exists():
        print(f"  SLAC data not found at {slac_path}")
        return None

    with open(slac_path, 'rb') as f:
        slac_data = pickle.load(f)

    # ABTS extinction coefficient at 420nm
    epsilon_420 = 36000  # M⁻¹cm⁻¹
    path_length = 0.5    # cm (96-well plate)

    sample = MultiConditionSample(
        mechanism='michaelis_menten_irreversible',
        mechanism_idx=0,
        energy_params={},
    )

    for temp in sorted(slac_data.keys()):
        slac_sample = slac_data[temp]
        for trace in slac_sample.traces:
            product_conc = trace.absorbance / (epsilon_420 * path_length) * 1000  # mM
            S0 = trace.substrate_conc  # mM
            P_formed = product_conc - product_conc[0]
            S = np.maximum(S0 - P_formed, 0)

            sample.add_trajectory(
                conditions={'S0': S0, 'E0': 1e-3, 'T': temp + 273.15},
                t=trace.time.astype(float),
                concentrations={'S': S, 'P': P_formed},
            )

    info = {
        'name': 'SLAC Laccase',
        'source': 'EnzymeML/DaRUS DOI:10.18419/darus-2096',
        'enzyme': 'Small laccase (S. coelicolor)',
        'ec': 'EC 1.10.3.2',
        'expected': 'michaelis_menten_irreversible',
        'n_traces': sample.n_conditions,
        'description': f'5 temperatures × 10 [S], {sample.n_conditions} traces',
    }
    return sample, info


def load_icekat_data(base_dir: Path):
    """Load ICEKAT SIRT1 deacetylase data."""
    import pandas as pd

    csv_path = base_dir / "data" / "real" / "icekat_test.csv"
    if not csv_path.exists():
        print(f"  ICEKAT data not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    t = df['Time (s)'].values.astype(float)

    sample = MultiConditionSample(
        mechanism='michaelis_menten_irreversible',
        mechanism_idx=0,
        energy_params={},
    )

    # Column format: "0 uM", "2.5 uM", "5 uM", etc.
    for col in df.columns[1:]:
        S0_uM = float(col.replace(' uM', ''))
        S0_mM = S0_uM / 1000.0  # Convert µM to mM

        # ICEKAT data is absorbance (product formation)
        # Absorbance decreases = substrate consumed (for deacetylation assay)
        absorbance = df[col].values.astype(float)

        # For SIRT1 assay, signal is proportional to product
        # Treat as product formation; infer substrate depletion
        # Normalize: treat initial absorbance as proportional to S0
        if S0_mM <= 0:
            continue  # Skip 0 µM control

        # Convert absorbance to approximate substrate concentration
        # Initial rate is proportional to [S], signal decays as substrate is consumed
        # For mechanism classification, the shape matters more than absolute values
        P = absorbance  # Product proxy
        P_formed = P - P[0]  # Relative product formation
        S = np.maximum(S0_mM - P_formed * S0_mM / (abs(P[-1] - P[0]) + 1e-10) * 0.5, 0)

        sample.add_trajectory(
            conditions={'S0': S0_mM, 'E0': 1e-3},
            t=t,
            concentrations={'S': S, 'P': P_formed},
        )

    info = {
        'name': 'ICEKAT SIRT1',
        'source': 'github.com/SmithLabMCW/icekat',
        'enzyme': 'SIRT1 deacetylase',
        'ec': 'EC 3.5.1.-',
        'expected': 'michaelis_menten_irreversible',
        'n_traces': sample.n_conditions,
        'description': f'8 [S] (2.5-500 µM), {sample.n_conditions} traces',
    }
    return sample, info


def load_abts_laccase_data(base_dir: Path):
    """Load ABTS laccase data from Lauterbach_2022 Scenario 4."""
    data_dir = Path("/tmp/enzymeml_extract/Scenario4_ABTS_Measurement_Ngubane/data")

    # Also check if extracted data exists at the expected location
    if not data_dir.exists():
        # Try extracting from OMEX
        omex_path = base_dir / "data" / "real" / "Lauterbach_2022" / "Scenario4" / "ABTS_Measurement_Ngubane.omex"
        if omex_path.exists():
            import zipfile
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(omex_path) as z:
                z.extractall(data_dir.parent)
        else:
            print(f"  ABTS laccase data not found")
            return None

    if not data_dir.exists():
        print(f"  ABTS laccase data directory not found: {data_dir}")
        return None

    sample = MultiConditionSample(
        mechanism='michaelis_menten_irreversible',
        mechanism_idx=0,
        energy_params={},
    )

    csv_files = sorted(data_dir.glob("m*.csv"), key=lambda p: int(p.stem[1:]))

    for csv_path in csv_files:
        # Format: time, replicate1, replicate2, replicate3
        # Values are substrate concentration in µmol/L
        data = np.genfromtxt(csv_path, delimiter=',')
        t = data[:, 0]  # seconds
        # Average replicates
        S_reps = data[:, 1:]  # µmol/L (µM)
        S_mean = np.nanmean(S_reps, axis=1)

        S0_uM = S_mean[0]
        S0_mM = S0_uM / 1000.0

        # Product formed = S0 - S(t)
        S_mM = S_mean / 1000.0  # Convert to mM
        P_mM = np.maximum(S0_mM - S_mM, 0)

        sample.add_trajectory(
            conditions={'S0': S0_mM, 'E0': 0.93e-3},  # 0.93 µM enzyme from XML
            t=t,
            concentrations={'S': S_mM, 'P': P_mM},
        )

    info = {
        'name': 'ABTS Laccase (T. pubescens)',
        'source': 'EnzymeML/Lauterbach_2022 Scenario 4',
        'enzyme': 'Laccase 2 (Trametes pubescens)',
        'ec': 'EC 1.10.3.2',
        'expected': 'michaelis_menten_irreversible',
        'n_traces': sample.n_conditions,
        'description': f'9 [S] (6.5-149 µM), {sample.n_conditions} traces, 3 replicates averaged',
    }
    return sample, info


# ═══════════════════════════════════════════════════════════════════════
# CONVERSION: MultiConditionSample → TACTIC model input tensors
# ═══════════════════════════════════════════════════════════════════════

def sample_to_tactic_input(sample: MultiConditionSample, device: torch.device,
                            n_timepoints: int = 20, max_conditions: int = 20):
    """Convert a MultiConditionSample to TACTIC model input tensors."""
    all_trajectories = []
    all_conditions = []
    all_derived = []

    for traj in sample.trajectories:
        t = traj['t']
        S = traj['concentrations']['S']
        P = traj['concentrations'].get('P', np.zeros_like(S))
        conds = traj['conditions']
        S0 = conds['S0']

        # Resample to fixed timepoints
        t_new = np.linspace(t.min(), t.max(), n_timepoints)
        S_new = np.interp(t_new, t, S)
        P_new = np.interp(t_new, t, P)

        # Normalize time to [0, 1]
        t_norm = t_new / (t_new[-1] + 1e-8)

        # Compute rates
        dS_dt = np.gradient(S_new, t_new)
        dP_dt = np.gradient(P_new, t_new)

        # Trajectory features: [t_norm, S, P, dS/dt, dP/dt]
        traj_feat = np.stack([t_norm, S_new, P_new, dS_dt, dP_dt], axis=-1)
        all_trajectories.append(traj_feat)

        # Condition features [S0, I0, B0, P0, E0, T_norm, pH_norm, type]
        cond = np.zeros(8)
        cond[0] = np.log10(max(S0, 1e-9))
        cond[1] = -9.0  # No inhibitor
        cond[2] = -9.0  # No second substrate
        cond[3] = -9.0  # No initial product
        cond[4] = np.log10(max(conds.get('E0', 1e-3), 1e-12))
        T = conds.get('T', 298.15)
        cond[5] = (T - 298.15) / 20.0 if T > 200 else (T - 25.0) / 20.0  # Handle K vs C
        cond[6] = 0.0  # pH unknown, use neutral
        cond[7] = 0.0  # Substrate variation
        all_conditions.append(cond)

        # Derived features
        derived = np.zeros(8)
        if len(t_new) > 1:
            v0 = abs(dS_dt[0])
            derived[0] = np.log10(max(v0, 1e-10))
        S_frac = S_new / (S_new[0] + 1e-10)
        half_idx = np.searchsorted(-S_frac, -0.5)
        if half_idx < len(t_new):
            derived[1] = np.log10(max(t_new[half_idx], 1.0))
        if len(t_new) > 2:
            v_start = abs(dS_dt[0])
            v_end = abs(dS_dt[-1])
            derived[2] = v_end / (v_start + 1e-10)
        derived[3] = 1 - S_new[-1] / (S_new[0] + 1e-10) if S_new[0] > 1e-10 else 0
        derived[4] = np.log10(max(S0, 1e-9))
        all_derived.append(derived)

    n_total = len(all_trajectories)
    n_conditions = min(n_total, max_conditions)

    trajectories = np.zeros((max_conditions, n_timepoints, 5))
    conditions = np.zeros((max_conditions, 8))
    derived_features = np.zeros((max_conditions, 8))
    condition_mask = np.ones(max_conditions, dtype=bool)

    for i in range(n_conditions):
        trajectories[i] = all_trajectories[i]
        conditions[i] = all_conditions[i]
        derived_features[i] = all_derived[i]
        condition_mask[i] = False

    return {
        'trajectories': torch.tensor(trajectories, dtype=torch.float32).unsqueeze(0).to(device),
        'conditions': torch.tensor(conditions, dtype=torch.float32).unsqueeze(0).to(device),
        'derived_features': torch.tensor(derived_features, dtype=torch.float32).unsqueeze(0).to(device),
        'condition_mask': torch.tensor(condition_mask, dtype=torch.bool).unsqueeze(0).to(device),
        'n_conditions_used': n_conditions,
        'n_conditions_total': n_total,
    }


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE: TACTIC and Classical AIC
# ═══════════════════════════════════════════════════════════════════════

def run_classical_aic(sample: MultiConditionSample) -> dict:
    """Run classical AIC model selection."""
    from classical_baseline import ClassicalModelSelector

    selector = ClassicalModelSelector(criterion='aic')

    start = time.perf_counter()
    pred_mech, all_results = selector.predict(sample)
    elapsed = time.perf_counter() - start

    fit_results = {}
    for mech, result in all_results.items():
        fit_results[mech] = {
            'aic': result.aic if result.aic != np.inf else None,
            'bic': result.bic if result.bic != np.inf else None,
            'ss_res': result.ss_res if result.ss_res != np.inf else None,
            'n_params': result.n_params,
            'success': result.success,
            'message': result.message,
        }

    return {
        'predicted': pred_mech,
        'elapsed_s': elapsed,
        'fit_results': fit_results,
    }


@torch.no_grad()
def run_tactic_inference(model, inputs: dict, device: torch.device) -> dict:
    """Run TACTIC inference."""
    start = time.perf_counter()

    output = model(
        inputs['trajectories'],
        inputs['conditions'],
        derived_features=inputs['derived_features'],
        condition_mask=inputs['condition_mask'],
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    logits = output['logits']
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_mech = MECHANISMS[pred_idx]

    return {
        'predicted': pred_mech,
        'confidence': float(probs[pred_idx]),
        'all_probs': {MECHANISMS[i]: float(probs[i]) for i in range(10)},
        'elapsed_ms': elapsed * 1000,
    }


def load_model(checkpoint_path: Path, device: torch.device, version: str = 'v3'):
    """Load trained TACTIC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if version == 'v1':
        model = create_basic_multi_condition_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )
    elif version == 'v2':
        model = create_multi_condition_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )
    else:
        model = create_multi_task_model(
            d_model=128, n_heads=4, n_traj_layers=2, n_cross_layers=3,
            n_mechanisms=10, dropout=0.0,
        )

    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# EVALUATE ONE DATASET
# ═══════════════════════════════════════════════════════════════════════

def evaluate_dataset(name: str, sample: MultiConditionSample, info: dict,
                     model, device: torch.device) -> dict:
    """Evaluate one dataset with both TACTIC and Classical AIC."""
    expected = info['expected']

    print(f"\n{'─' * 70}")
    print(f"Dataset: {info['name']}")
    print(f"Enzyme:  {info['enzyme']} ({info['ec']})")
    print(f"Source:  {info['source']}")
    print(f"Data:    {info['description']}")
    print(f"Expected mechanism: {expected}")
    print(f"{'─' * 70}")

    # TACTIC inference
    print("\n  TACTIC Inference:")
    inputs = sample_to_tactic_input(sample, device, max_conditions=20)
    tactic_result = run_tactic_inference(model, inputs, device)

    tactic_correct = tactic_result['predicted'] == expected
    print(f"    Predicted:  {tactic_result['predicted']}")
    print(f"    Confidence: {tactic_result['confidence']*100:.1f}%")
    print(f"    Time:       {tactic_result['elapsed_ms']:.2f} ms")
    print(f"    Result:     {'CORRECT' if tactic_correct else 'WRONG'}")

    # Top-3
    sorted_probs = sorted(tactic_result['all_probs'].items(), key=lambda x: -x[1])
    print("    Top-3:")
    for mech, prob in sorted_probs[:3]:
        marker = " <-- EXPECTED" if mech == expected else ""
        print(f"      {mech:35s}: {prob*100:5.1f}%{marker}")

    # Classical AIC
    print("\n  Classical AIC:")
    classical_result = run_classical_aic(sample)

    classical_correct = classical_result['predicted'] == expected
    print(f"    Predicted:  {classical_result['predicted']}")
    print(f"    Time:       {classical_result['elapsed_s']:.2f} s")
    print(f"    Result:     {'CORRECT' if classical_correct else 'WRONG'}")

    # AIC ranking
    valid_fits = {m: r for m, r in classical_result['fit_results'].items()
                  if r['aic'] is not None}
    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1]['aic'])
    print("    AIC ranking:")
    for i, (mech, result) in enumerate(sorted_fits[:5], 1):
        marker = " <-- EXPECTED" if mech == expected else ""
        marker += " <-- SELECTED" if mech == classical_result['predicted'] else ""
        print(f"      {i}. {mech:35s} AIC={result['aic']:10.1f}{marker}")

    speedup = classical_result['elapsed_s'] / (tactic_result['elapsed_ms'] / 1000) if tactic_result['elapsed_ms'] > 0 else 0

    return {
        'info': info,
        'tactic': {
            'predicted': tactic_result['predicted'],
            'confidence': tactic_result['confidence'],
            'correct': tactic_correct,
            'elapsed_ms': tactic_result['elapsed_ms'],
            'all_probs': tactic_result['all_probs'],
        },
        'classical': {
            'predicted': classical_result['predicted'],
            'correct': classical_correct,
            'elapsed_s': classical_result['elapsed_s'],
            'aic_ranking': [(m, r['aic']) for m, r in sorted_fits[:5]],
        },
        'speedup': speedup,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Real Data Evaluation: TACTIC vs AIC')
    parser.add_argument('--version', type=str, default='v3', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save-results', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/{args.version}/best_model.pt'

    print("=" * 70)
    print(f"EXPERIMENT 8: Real Data Evaluation ({args.version.upper()})")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.version}")

    # Load model
    checkpoint_path = base_dir / args.checkpoint
    model = load_model(checkpoint_path, device, version=args.version)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ── Load all datasets ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    datasets = []

    # 1. SLAC Laccase
    print("\n1. Loading SLAC Laccase...")
    result = load_slac_data(base_dir)
    if result:
        datasets.append(result)
        print(f"   Loaded: {result[1]['n_traces']} traces")

    # 2. ICEKAT SIRT1
    print("2. Loading ICEKAT SIRT1...")
    result = load_icekat_data(base_dir)
    if result:
        datasets.append(result)
        print(f"   Loaded: {result[1]['n_traces']} traces")

    # 3. ABTS Laccase (Lauterbach_2022)
    print("3. Loading ABTS Laccase (Lauterbach_2022)...")
    result = load_abts_laccase_data(base_dir)
    if result:
        datasets.append(result)
        print(f"   Loaded: {result[1]['n_traces']} traces")

    print(f"\nTotal datasets loaded: {len(datasets)}")

    # ── Evaluate each dataset ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    all_results = []
    for sample, info in datasets:
        result = evaluate_dataset(info['name'], sample, info, model, device)
        all_results.append(result)

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: TACTIC vs CLASSICAL AIC ON REAL DATA")
    print("=" * 70)

    print(f"\n{'Dataset':<30} {'Expected':<15} {'TACTIC':<20} {'Classical AIC':<20} {'Speed':<10}")
    print("-" * 95)

    tactic_correct_total = 0
    classical_correct_total = 0

    for r in all_results:
        name = r['info']['name'][:29]
        expected = 'MM irrev'
        tactic_str = f"{'OK' if r['tactic']['correct'] else 'WRONG'} ({r['tactic']['confidence']*100:.0f}%)"
        classical_str = 'OK' if r['classical']['correct'] else 'WRONG'
        speed_str = f"{r['speedup']:.0f}x"

        if r['tactic']['correct']:
            tactic_correct_total += 1
        if r['classical']['correct']:
            classical_correct_total += 1

        print(f"{name:<30} {expected:<15} {tactic_str:<20} {classical_str:<20} {speed_str:<10}")

    n = len(all_results)
    print("-" * 95)
    print(f"{'TOTAL':<30} {'':<15} {tactic_correct_total}/{n} correct{'':<10} {classical_correct_total}/{n} correct")

    tactic_avg_ms = np.mean([r['tactic']['elapsed_ms'] for r in all_results])
    classical_avg_s = np.mean([r['classical']['elapsed_s'] for r in all_results])
    overall_speedup = classical_avg_s / (tactic_avg_ms / 1000) if tactic_avg_ms > 0 else 0

    print(f"\nAverage inference time: TACTIC={tactic_avg_ms:.1f}ms, Classical={classical_avg_s:.1f}s")
    print(f"Average speedup: {overall_speedup:.0f}x")

    # ── Save results ────────────────────────────────────────────────────
    if args.save_results:
        results_dir = base_dir / "results" / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"real_data_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            'version': args.version,
            'n_datasets': len(all_results),
            'datasets': [],
            'summary': {
                'tactic_correct': tactic_correct_total,
                'classical_correct': classical_correct_total,
                'total': n,
                'tactic_avg_ms': tactic_avg_ms,
                'classical_avg_s': classical_avg_s,
                'overall_speedup': overall_speedup,
            },
        }

        for r in all_results:
            dataset_result = {
                'name': r['info']['name'],
                'enzyme': r['info']['enzyme'],
                'source': r['info']['source'],
                'expected': r['info']['expected'],
                'n_traces': r['info']['n_traces'],
                'tactic': r['tactic'],
                'classical': {
                    'predicted': r['classical']['predicted'],
                    'correct': r['classical']['correct'],
                    'elapsed_s': r['classical']['elapsed_s'],
                },
                'speedup': r['speedup'],
            }
            output['datasets'].append(dataset_result)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
