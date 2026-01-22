#!/usr/bin/env python
"""
Evaluate TACTIC model on real enzyme kinetics data.

This script:
1. Loads parsed SLAC data
2. Converts to TACTIC input format
3. Runs inference
4. Reports mechanism predictions
"""

import numpy as np
import torch
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from parse_slac_data import SLACSample, KineticTrace

from tactic_kinetics.models.multi_condition_classifier import (
    MultiConditionClassifier,
    MultiTaskClassifier,
    create_multi_condition_model,
    create_multi_task_model,
)

MECHANISM_NAMES = [
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


class RealDataPreprocessor:
    """Convert real kinetics data to TACTIC input format."""

    def __init__(self, n_timepoints: int = 20, n_traj_features: int = 5,
                 n_condition_features: int = 8, n_derived_features: int = 8):
        self.n_timepoints = n_timepoints
        self.n_traj_features = n_traj_features
        self.n_condition_features = n_condition_features
        self.n_derived_features = n_derived_features

        # Normalization parameters (should match training data)
        # These are approximate values - ideally load from training
        self.conc_mean = 0.5
        self.conc_std = 5.0
        self.time_max = 1000.0

    def preprocess_slac(self, slac_samples: dict, max_conditions: int = 20) -> dict:
        """
        Convert SLAC data to TACTIC format.

        SLAC has multiple substrate concentrations at different temperatures.
        We treat each [S] as a separate condition.

        Args:
            slac_samples: Dict[temperature -> SLACSample]
            max_conditions: Maximum conditions per sample

        Returns:
            Dict with 'trajectories', 'conditions', 'derived_features', 'condition_mask'
        """
        # For SLAC, we'll create one sample combining data across temperatures
        # Each trace (substrate concentration) becomes one condition

        all_trajectories = []
        all_conditions = []
        all_derived = []

        # Collect all traces
        for temp, sample in sorted(slac_samples.items()):
            for trace in sample.traces:
                # Resample trajectory to fixed timepoints
                t_resampled, conc_resampled = self._resample(trace.time, trace.absorbance)

                # Convert absorbance to "concentration" proxy
                # Absorbance ∝ product concentration for ABTS oxidation
                # We'll treat it as product formation
                S0 = trace.absorbance[0]  # Initial absorbance as proxy for initial state
                P = conc_resampled  # Absorbance as product proxy
                S = S0 - (P - P[0])  # Substrate = initial - (product formed)
                S = np.maximum(S, 0)  # Ensure non-negative

                # Compute trajectory features: [t_norm, S_norm, P_norm, dS/dt, dP/dt]
                traj_features = self._compute_trajectory_features(t_resampled, S, P)
                all_trajectories.append(traj_features)

                # Condition features
                cond = self._encode_conditions({
                    'S0': trace.substrate_conc,
                    'T': trace.temperature + 273.15,  # Convert to Kelvin
                    'pH': 3.0,  # From filename
                    'E0': 1e-6,  # Unknown, use default
                    'I0': 0.0,  # No inhibitor
                })
                all_conditions.append(cond)

                # Derived features
                derived = self._compute_derived_features(t_resampled, S, P, trace.substrate_conc)
                all_derived.append(derived)

        # Truncate or pad to max_conditions
        n_conditions = min(len(all_trajectories), max_conditions)

        trajectories = np.zeros((max_conditions, self.n_timepoints, self.n_traj_features))
        conditions = np.zeros((max_conditions, self.n_condition_features))
        derived_features = np.zeros((max_conditions, self.n_derived_features))
        condition_mask = np.ones(max_conditions, dtype=bool)  # True = invalid

        for i in range(n_conditions):
            trajectories[i] = all_trajectories[i]
            conditions[i] = all_conditions[i]
            derived_features[i] = all_derived[i]
            condition_mask[i] = False  # Valid

        return {
            'trajectories': torch.tensor(trajectories, dtype=torch.float32).unsqueeze(0),
            'conditions': torch.tensor(conditions, dtype=torch.float32).unsqueeze(0),
            'derived_features': torch.tensor(derived_features, dtype=torch.float32).unsqueeze(0),
            'condition_mask': torch.tensor(condition_mask, dtype=torch.bool).unsqueeze(0),
        }

    def _resample(self, t: np.ndarray, y: np.ndarray) -> tuple:
        """Resample to fixed number of timepoints."""
        from scipy.interpolate import interp1d

        f = interp1d(t, y, kind='linear', fill_value='extrapolate')
        t_new = np.linspace(t.min(), t.max(), self.n_timepoints)
        y_new = f(t_new)

        return t_new, y_new

    def _compute_trajectory_features(self, t: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute trajectory features matching TACTIC format.

        Features: [t_norm, S_norm, P_norm, dS/dt_norm, dP/dt_norm]
        """
        # Normalize time
        t_norm = t / self.time_max

        # Normalize concentrations
        S_norm = (S - self.conc_mean) / self.conc_std
        P_norm = (P - self.conc_mean) / self.conc_std

        # Compute rates
        dS_dt = np.gradient(S, t)
        dP_dt = np.gradient(P, t)

        # Normalize rates
        rate_scale = self.conc_std / self.time_max + 1e-8
        dS_dt_norm = dS_dt / rate_scale
        dP_dt_norm = dP_dt / rate_scale

        # Stack features
        features = np.stack([t_norm, S_norm, P_norm, dS_dt_norm, dP_dt_norm], axis=-1)

        return features

    def _encode_conditions(self, conditions: dict) -> np.ndarray:
        """Encode experimental conditions."""
        features = np.zeros(self.n_condition_features)

        # Primary substrate (log scale)
        S0 = conditions.get('S0', 1.0)
        features[0] = np.log10(max(S0, 1e-9))

        # Inhibitor (0 for no inhibitor)
        I0 = conditions.get('I0', 0)
        features[1] = np.log10(max(I0, 1e-9)) if I0 > 0 else -9.0

        # Second substrate (0 for single substrate)
        features[2] = -9.0

        # Product
        features[3] = -9.0

        # Enzyme
        E0 = conditions.get('E0', 1e-6)
        features[4] = np.log10(max(E0, 1e-12))

        # Temperature (normalized)
        T = conditions.get('T', 298.15)
        features[5] = (T - 298.15) / 20.0

        # pH (normalized)
        pH = conditions.get('pH', 7.0)
        features[6] = (pH - 7.0) / 2.0

        # Condition type
        features[7] = 0.0

        return features

    def _compute_derived_features(self, t: np.ndarray, S: np.ndarray, P: np.ndarray,
                                   S0: float) -> np.ndarray:
        """Compute derived kinetic features."""
        features = np.zeros(self.n_derived_features)

        # Initial rate (v0)
        if len(t) > 1:
            v0 = -np.gradient(S, t)[0]
            features[0] = np.log10(max(abs(v0), 1e-10))

        # Half-life estimate
        S_frac = S / (S[0] + 1e-10)
        half_idx = np.searchsorted(-S_frac, -0.5)
        if half_idx < len(t):
            t_half = t[half_idx]
            features[1] = np.log10(max(t_half, 1.0))

        # Rate ratio (v_end / v_start)
        if len(t) > 2:
            v_start = abs(np.gradient(S, t)[0])
            v_end = abs(np.gradient(S, t)[-1])
            features[2] = v_end / (v_start + 1e-10)

        # Final conversion
        features[3] = 1 - S[-1] / (S[0] + 1e-10)

        # Substrate utilization
        features[4] = np.log10(max(S0, 1e-9))

        # Other features (zeros for now)
        features[5:] = 0.0

        return features


def load_model(checkpoint_path: Path, device: torch.device, multi_task: bool = True):
    """Load trained TACTIC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if multi_task:
        model = create_multi_task_model(
            d_model=128,
            n_heads=4,
            n_traj_layers=2,
            n_cross_layers=3,
            n_mechanisms=10,
            dropout=0.0,  # No dropout for inference
        )
    else:
        model = create_multi_condition_model(
            d_model=128,
            n_heads=4,
            n_traj_layers=2,
            n_cross_layers=3,
            n_mechanisms=10,
            dropout=0.0,
        )

    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate_slac(model, slac_data: dict, device: torch.device):
    """
    Evaluate model on SLAC laccase data.

    Expected: michaelis_menten_irreversible
    """
    preprocessor = RealDataPreprocessor()
    inputs = preprocessor.preprocess_slac(slac_data)

    # Move to device
    trajectories = inputs['trajectories'].to(device)
    conditions = inputs['conditions'].to(device)
    derived_features = inputs['derived_features'].to(device)
    condition_mask = inputs['condition_mask'].to(device)

    # Run inference
    output = model(
        trajectories,
        conditions,
        derived_features=derived_features,
        condition_mask=condition_mask,
    )

    logits = output['logits']
    probs = torch.softmax(logits, dim=-1)

    # Get predictions
    pred_idx = logits.argmax(dim=-1).item()
    pred_mechanism = MECHANISM_NAMES[pred_idx]
    confidence = probs[0, pred_idx].item()

    # Print results
    print("\n" + "="*70)
    print("SLAC LACCASE EVALUATION RESULTS")
    print("="*70)
    print(f"Expected mechanism: michaelis_menten_irreversible")
    print(f"Predicted mechanism: {pred_mechanism}")
    print(f"Confidence: {confidence*100:.1f}%")
    print()

    # Print all probabilities
    print("All mechanism probabilities:")
    sorted_probs = sorted(enumerate(probs[0].cpu().numpy()), key=lambda x: -x[1])
    for idx, prob in sorted_probs:
        marker = " <-- EXPECTED" if MECHANISM_NAMES[idx] == 'michaelis_menten_irreversible' else ""
        marker += " <-- PREDICTED" if idx == pred_idx else ""
        print(f"  {MECHANISM_NAMES[idx]:35s}: {prob*100:5.1f}%{marker}")

    # Check if correct
    correct = pred_mechanism == 'michaelis_menten_irreversible'
    print()
    print(f"Result: {'CORRECT' if correct else 'INCORRECT'}")

    return {
        'predicted': pred_mechanism,
        'expected': 'michaelis_menten_irreversible',
        'confidence': confidence,
        'correct': correct,
        'all_probs': {MECHANISM_NAMES[i]: probs[0, i].item() for i in range(10)},
    }


def main():
    data_dir = Path(__file__).parent.parent / "data" / "real"
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"

    print("="*70)
    print("TACTIC Real Data Evaluation")
    print("="*70)

    # Check for SLAC data
    slac_path = data_dir / "slac_parsed.pkl"
    if not slac_path.exists():
        print("SLAC data not found. Run parse_slac_data.py first.")
        return

    # Load SLAC data
    print(f"Loading SLAC data from: {slac_path}")
    with open(slac_path, 'rb') as f:
        slac_data = pickle.load(f)
    print(f"Loaded {len(slac_data)} temperature conditions")

    # Find model checkpoint
    checkpoint_path = checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found: {checkpoint_path}")
        print("Train a model first with: python train.py --multi-task")
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device, multi_task=True)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Evaluate on SLAC
    results = evaluate_slac(model, slac_data, device)

    return results


if __name__ == "__main__":
    main()
