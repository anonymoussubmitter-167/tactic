"""
Dataset and utilities for multi-condition training.

Handles conversion from MultiConditionSample to tensors suitable
for the MultiConditionClassifier.
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .multi_condition_generator import MultiConditionSample


@dataclass
class MultiConditionDatasetConfig:
    """Configuration for multi-condition dataset."""
    max_conditions: int = 20  # Maximum number of conditions per sample (increased)
    n_timepoints: int = 20  # Fixed number of timepoints
    n_condition_features: int = 8  # Number of condition features
    n_trajectory_features: int = 9  # Features per timepoint: t, S, P, dS/dt, dP/dt + 4 derived
    n_derived_features: int = 8  # Per-trajectory derived features (v0, t_half, etc.)
    n_kinetic_params: int = 2  # Per-condition kinetic params: v0_norm (proxy for Vmax_app), Km_app_est
    n_pattern_features: int = 4  # Cross-condition pattern: dVmax/dI, dKm/dI, dVmax/dS, dKm/dS


class MultiConditionDataset(Dataset):
    """
    Dataset for multi-condition samples.

    Each sample contains multiple trajectories from the same enzyme
    measured under different conditions.
    """

    def __init__(
        self,
        samples: List[MultiConditionSample],
        config: Optional[MultiConditionDatasetConfig] = None,
    ):
        self.config = config or MultiConditionDatasetConfig()
        self.samples = samples

        # Compute normalization statistics
        self._compute_stats()

    def _compute_stats(self):
        """Compute normalization statistics from the dataset."""
        all_conc = []
        all_times = []

        for sample in self.samples:
            for traj in sample.trajectories:
                all_times.extend(traj['t'].tolist())
                for species, conc in traj['concentrations'].items():
                    all_conc.extend(conc.tolist())

        self.conc_mean = np.mean(all_conc)
        self.conc_std = np.std(all_conc) + 1e-8
        self.time_max = np.max(all_times) + 1e-8

        print(f"Dataset stats: conc_mean={self.conc_mean:.4f}, conc_std={self.conc_std:.4f}, time_max={self.time_max:.2f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.samples[idx]

        # Process trajectories
        trajectories = []
        conditions_list = []
        derived_features_list = []

        for traj_data in sample.trajectories[:self.config.max_conditions]:
            concentrations = traj_data['concentrations']
            t = traj_data['t']
            conditions = traj_data['conditions']

            # Get substrate and product trajectories
            S, P = self._get_substrate_product(concentrations, sample.mechanism)

            # Interpolate to fixed number of timepoints if needed
            if len(t) != self.config.n_timepoints:
                t_new = np.linspace(t[0], t[-1], self.config.n_timepoints)
                S = np.interp(t_new, t, S)
                P = np.interp(t_new, t, P)
                t = t_new

            # Compute rates (d_conc/dt)
            dS_dt = np.gradient(S, t)
            dP_dt = np.gradient(P, t)

            # Normalize
            t_norm = t / self.time_max
            S_norm = (S - self.conc_mean) / self.conc_std
            P_norm = (P - self.conc_mean) / self.conc_std
            dS_dt_norm = dS_dt / (self.conc_std / self.time_max + 1e-8)
            dP_dt_norm = dP_dt / (self.conc_std / self.time_max + 1e-8)

            # Stack trajectory features: (n_timepoints, 5)
            # Features: t_norm, S_norm, P_norm, dS/dt_norm, dP/dt_norm
            traj_features = np.stack([t_norm, S_norm, P_norm, dS_dt_norm, dP_dt_norm], axis=-1)
            trajectories.append(traj_features)

            # Extract condition features
            cond_features = self._extract_condition_features(conditions)
            conditions_list.append(cond_features)

            # Compute derived kinetic features
            derived = self._compute_derived_features(t, S, P, conditions)
            derived_features_list.append(derived)

        # Pad to max_conditions
        n_actual = len(trajectories)
        n_pad = self.config.max_conditions - n_actual

        if n_pad > 0:
            pad_traj = np.zeros((self.config.n_timepoints, 5))  # Now 5 features
            pad_cond = np.zeros(self.config.n_condition_features)
            pad_derived = np.zeros(self.config.n_derived_features)
            for _ in range(n_pad):
                trajectories.append(pad_traj)
                conditions_list.append(pad_cond)
                derived_features_list.append(pad_derived)

        # Create mask (True = invalid/padded)
        condition_mask = np.array([False] * n_actual + [True] * n_pad)

        # Convert to tensors
        trajectories_tensor = torch.tensor(np.stack(trajectories), dtype=torch.float32)
        conditions_tensor = torch.tensor(np.stack(conditions_list), dtype=torch.float32)
        derived_tensor = torch.tensor(np.stack(derived_features_list), dtype=torch.float32)
        condition_mask_tensor = torch.tensor(condition_mask, dtype=torch.bool)

        # Compute auxiliary targets for multi-task learning
        kinetic_params, param_pattern = self._compute_auxiliary_targets(
            sample, trajectories, conditions_list, n_actual
        )

        # Pad kinetic params
        if n_pad > 0:
            pad_kinetic = np.zeros(self.config.n_kinetic_params)
            for _ in range(n_pad):
                kinetic_params.append(pad_kinetic)

        kinetic_params_tensor = torch.tensor(np.stack(kinetic_params), dtype=torch.float32)
        param_pattern_tensor = torch.tensor(param_pattern, dtype=torch.float32)

        return {
            'trajectories': trajectories_tensor,  # (max_conditions, n_timepoints, 5)
            'conditions': conditions_tensor,  # (max_conditions, n_condition_features)
            'derived_features': derived_tensor,  # (max_conditions, n_derived_features)
            'condition_mask': condition_mask_tensor,  # (max_conditions,)
            'mechanism_idx': torch.tensor(sample.mechanism_idx, dtype=torch.long),
            'mechanism': sample.mechanism,
            'n_conditions': n_actual,
            # Auxiliary targets for multi-task learning
            'kinetic_params': kinetic_params_tensor,  # (max_conditions, 2) - v0_norm, km_est
            'param_pattern': param_pattern_tensor,  # (4,) - parameter change slopes
        }

    def _get_substrate_product(self, concentrations: Dict[str, np.ndarray], mechanism: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get substrate and product trajectories.

        Returns (S, P) for single substrate mechanisms
        Returns (A, P) for bi-substrate mechanisms (A is primary substrate)
        """
        if 'S' in concentrations:
            S = concentrations['S']
            P = concentrations.get('P', np.zeros_like(S))
        elif 'A' in concentrations:
            # Bi-substrate: use A as primary, P as product
            S = concentrations['A']
            P = concentrations.get('P', np.zeros_like(S))
        else:
            # Fallback
            keys = list(concentrations.keys())
            S = concentrations[keys[0]]
            P = concentrations[keys[1]] if len(keys) > 1 else np.zeros_like(S)

        return S, P

    def _compute_derived_features(self, t: np.ndarray, S: np.ndarray, P: np.ndarray,
                                   conditions: Dict) -> np.ndarray:
        """
        Compute derived kinetic features that are diagnostic for mechanism discrimination.

        These features are what biochemists actually look at:
        - Initial rate (v0)
        - Time to half conversion
        - Final conversion
        - Rate ratio (late/early)
        - Curve shape deviation from simple exponential

        Features (8 total):
        0. v0_normalized - initial rate normalized by E0
        1. t_half_normalized - time to 50% conversion / t_max
        2. final_conversion - fraction of substrate converted
        3. rate_ratio - v_late / v_early (indicates product inhibition/reversibility)
        4. P_final_normalized - final product concentration
        5. mass_balance_error - should be ~0 for good data
        6. exponential_residual - deviation from simple exponential decay
        7. acceleration - second derivative at midpoint (curve shape)
        """
        features = np.zeros(self.config.n_derived_features)

        S0 = S[0] if len(S) > 0 else 1.0
        E0 = conditions.get('E0', 1e-3)

        # 1. Initial rate (v0) - slope at t=0
        if len(t) > 1:
            v0 = -np.gradient(S, t)[0]  # Negative because S decreases
            features[0] = v0 / (E0 + 1e-12)  # Normalize by enzyme concentration

        # 2. Time to 50% conversion
        if S0 > 0:
            S_frac = S / S0
            half_indices = np.where(S_frac <= 0.5)[0]
            if len(half_indices) > 0:
                t_half = t[half_indices[0]]
            else:
                t_half = t[-1]  # Didn't reach 50%
            features[1] = t_half / (t[-1] + 1e-8)

        # 3. Final conversion
        if S0 > 0:
            features[2] = 1 - S[-1] / S0

        # 4. Rate ratio (late phase vs early phase)
        if len(t) > 4:
            early_rate = -np.gradient(S, t)[1]  # Near start
            late_idx = len(t) * 3 // 4
            late_rate = -np.gradient(S, t)[late_idx]
            features[3] = late_rate / (early_rate + 1e-10)

        # 5. Final product (normalized)
        features[4] = P[-1] / (self.conc_std + 1e-8)

        # 6. Mass balance error
        if S0 > 0:
            features[5] = (S0 - S[-1] - P[-1]) / S0

        # 7. Exponential residual (deviation from simple kinetics)
        if S0 > 0 and len(t) > 2:
            # Simple exponential would be S = S0 * exp(-k*t)
            # Check deviation from this
            try:
                # Estimate k from t_half
                k_est = 0.693 / (features[1] * t[-1] + 1e-8)
                S_exp = S0 * np.exp(-k_est * t)
                residual = np.mean((S - S_exp) ** 2) / (S0 ** 2 + 1e-8)
                features[6] = np.clip(residual, 0, 1)
            except:
                features[6] = 0.0

        # 8. Acceleration at midpoint (second derivative)
        if len(t) > 4:
            mid_idx = len(t) // 2
            d2S_dt2 = np.gradient(np.gradient(S, t), t)
            features[7] = d2S_dt2[mid_idx] / (self.conc_std / (self.time_max ** 2) + 1e-8)

        return features

    def _compute_auxiliary_targets(
        self,
        sample,
        trajectories: List[np.ndarray],
        conditions_list: List[np.ndarray],
        n_actual: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute auxiliary targets for multi-task learning.

        Returns:
            kinetic_params: List of (2,) arrays - [v0_normalized, km_estimate] per condition
            param_pattern: (4,) array - [dVmax/dI, dKm/dI, dVmax/dA, dKm/dA] change slopes
        """
        kinetic_params = []
        v0_values = []
        conditions_raw = []

        for i, traj_data in enumerate(sample.trajectories[:n_actual]):
            t = traj_data['t']
            concentrations = traj_data['concentrations']
            cond = traj_data['conditions']

            S, P = self._get_substrate_product(concentrations, sample.mechanism)
            S0 = S[0] if len(S) > 0 else 1.0
            E0 = cond.get('E0', 1e-3)

            # Compute initial rate v0
            if len(t) > 1:
                v0 = -np.gradient(S, t)[0]  # Negative because S decreases
                v0 = max(v0, 1e-12)  # Ensure positive (handles noise/reversible)
            else:
                v0 = 1e-12

            # Normalize by E0 (this is proportional to Vmax_app * S0/(Km+S0))
            v0_norm = v0 / (E0 + 1e-12)
            v0_norm = max(v0_norm, 1e-10)  # Ensure positive for log

            # Estimate apparent Km using Michaelis-Menten approximation
            # v0 = Vmax_app * S0 / (Km_app + S0)
            # If we assume v0 at highest S0 ≈ Vmax_app, we can estimate Km
            # For now, use a simpler proxy: ratio of v0 to max possible rate
            km_estimate = S0 / (v0_norm + 1e-10) - S0 if v0_norm > 1e-10 else 1.0
            km_estimate = np.clip(km_estimate, 1e-6, 100)  # Reasonable bounds, ensure positive

            kinetic_params.append(np.array([
                np.log10(v0_norm),  # Log scale for v0 (guaranteed positive)
                np.log10(km_estimate),  # Log scale for Km estimate (guaranteed positive)
            ]))

            v0_values.append(v0_norm)
            conditions_raw.append(cond)

        # Compute parameter change pattern across conditions
        param_pattern = self._compute_param_change_pattern(
            v0_values, conditions_raw, sample.mechanism
        )

        # Final NaN check on kinetic_params
        kinetic_params = [np.nan_to_num(kp, nan=0.0, posinf=10.0, neginf=-10.0) for kp in kinetic_params]

        return kinetic_params, param_pattern

    def _compute_param_change_pattern(
        self,
        v0_values: List[float],
        conditions: List[Dict],
        mechanism: str
    ) -> np.ndarray:
        """
        Compute how kinetic parameters change across conditions.

        This is THE KEY discriminating feature:
        - Competitive: Km ↑ with [I], Vmax unchanged
        - Uncompetitive: Km ↓ with [I], Vmax ↓ with [I]
        - Mixed: Both change

        Returns (4,):
        0. dVmax/d[I] slope (0 for competitive, negative for uncompetitive)
        1. dKm/d[I] slope (positive for competitive, negative for uncompetitive)
        2. dVmax/d[A] or d[S] slope
        3. dKm/d[A] or d[S] slope
        """
        pattern = np.zeros(4)

        if len(v0_values) < 2:
            return pattern

        # Extract concentration values
        I_values = [c.get('I0', 0) for c in conditions]
        S_values = [c.get('S0', c.get('A0', 1.0)) for c in conditions]

        v0_arr = np.array(v0_values)

        # Compute slopes if there's variation in [I]
        I_arr = np.array(I_values)
        if I_arr.max() > I_arr.min() + 1e-10:
            # Fit linear regression: v0 = a*I + b
            try:
                slope_v_I = np.polyfit(I_arr, v0_arr, 1)[0]
                pattern[0] = np.clip(slope_v_I / (v0_arr.mean() + 1e-10), -10, 10)
            except:
                pass

            # For Km estimation with [I], look at how v0/S0 ratio changes
            # This is a proxy for Vmax/Km changes
            try:
                S_arr = np.array(S_values)
                v_over_S = v0_arr / (S_arr + 1e-10)
                slope_km_I = np.polyfit(I_arr, v_over_S, 1)[0]
                pattern[1] = np.clip(slope_km_I / (v_over_S.mean() + 1e-10), -10, 10)
            except:
                pass

        # Compute slopes with [S] or [A] variation
        S_arr = np.array(S_values)
        if S_arr.max() > S_arr.min() + 1e-10:
            try:
                slope_v_S = np.polyfit(S_arr, v0_arr, 1)[0]
                pattern[2] = np.clip(slope_v_S / (v0_arr.mean() + 1e-10), -10, 10)
            except:
                pass

            # Km proxy: at what [S] does v0 reach half-max?
            try:
                v0_max = v0_arr.max()
                half_max_idx = np.argmin(np.abs(v0_arr - v0_max / 2))
                pattern[3] = np.log10(max(S_arr[half_max_idx], 1e-10))
            except:
                pass

        # Final NaN/Inf check - replace with zeros
        pattern = np.nan_to_num(pattern, nan=0.0, posinf=10.0, neginf=-10.0)
        return pattern

    def _extract_condition_features(self, conditions: Dict) -> np.ndarray:
        """
        Extract and normalize condition features.

        Features (in order):
        0. log10(S0) or log10(A0) - primary substrate
        1. log10(I0) or 0 if no inhibitor
        2. log10(B0) or 0 if single substrate
        3. log10(P0) or 0 if no initial product
        4. log10(E0) - enzyme concentration
        5. (T - 298) / 20 - normalized temperature
        6. (pH - 7) / 2 - normalized pH
        7. condition type indicator (0=vary_S, 1=vary_I, 2=vary_both, etc.)
        """
        features = np.zeros(self.config.n_condition_features)

        # Primary substrate (S or A)
        S0 = conditions.get('S0', conditions.get('A0', 1.0))
        features[0] = np.log10(max(S0, 1e-9))

        # Inhibitor
        I0 = conditions.get('I0', 0)
        features[1] = np.log10(max(I0, 1e-9)) if I0 > 0 else -9.0

        # Second substrate (B)
        B0 = conditions.get('B0', 0)
        features[2] = np.log10(max(B0, 1e-9)) if B0 > 0 else -9.0

        # Product
        P0 = conditions.get('P0', 0)
        features[3] = np.log10(max(P0, 1e-9)) if P0 > 0 else -9.0

        # Enzyme
        E0 = conditions.get('E0', 1e-3)
        features[4] = np.log10(max(E0, 1e-12))

        # Temperature
        T = conditions.get('T', 298.15)
        features[5] = (T - 298.15) / 20.0

        # pH
        pH = conditions.get('pH', 7.0)
        features[6] = (pH - 7.0) / 2.0

        # Condition type indicator (which variable is being varied)
        # This helps the model understand the experimental design
        features[7] = 0.0  # Default: substrate variation

        return features


def multi_condition_collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    """
    Collate function for multi-condition batches.

    Handles variable number of conditions per sample by padding.
    """
    trajectories = torch.stack([item['trajectories'] for item in batch])
    conditions = torch.stack([item['conditions'] for item in batch])
    derived_features = torch.stack([item['derived_features'] for item in batch])
    condition_masks = torch.stack([item['condition_mask'] for item in batch])
    mechanism_idx = torch.stack([item['mechanism_idx'] for item in batch])
    kinetic_params = torch.stack([item['kinetic_params'] for item in batch])
    param_pattern = torch.stack([item['param_pattern'] for item in batch])

    return {
        'trajectories': trajectories,  # (batch, max_conditions, n_timepoints, 5)
        'conditions': conditions,  # (batch, max_conditions, n_condition_features)
        'derived_features': derived_features,  # (batch, max_conditions, n_derived_features)
        'condition_mask': condition_masks,  # (batch, max_conditions)
        'mechanism_idx': mechanism_idx,  # (batch,)
        # Auxiliary targets for multi-task learning
        'kinetic_params': kinetic_params,  # (batch, max_conditions, 2)
        'param_pattern': param_pattern,  # (batch, 4)
    }


def create_multi_condition_dataloaders(
    train_samples: List[MultiConditionSample],
    val_samples: List[MultiConditionSample],
    batch_size: int = 32,
    num_workers: int = 4,
    config: Optional[MultiConditionDatasetConfig] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    train_dataset = MultiConditionDataset(train_samples, config)
    val_dataset = MultiConditionDataset(val_samples, config)

    # Share normalization stats
    val_dataset.conc_mean = train_dataset.conc_mean
    val_dataset.conc_std = train_dataset.conc_std
    val_dataset.time_max = train_dataset.time_max

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=multi_condition_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multi_condition_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def generate_multi_condition_dataset(
    n_samples_per_mechanism: int = 1000,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[MultiConditionSample], List[MultiConditionSample]]:
    """
    Generate train and validation multi-condition datasets.

    Args:
        n_samples_per_mechanism: Number of samples per mechanism
        val_fraction: Fraction for validation
        seed: Random seed

    Returns:
        (train_samples, val_samples)
    """
    from .multi_condition_generator import MultiConditionGenerator, MultiConditionConfig

    config = MultiConditionConfig(
        n_conditions_per_sample=20,  # Increased for better discrimination
        n_timepoints=20,
        noise_level=0.03,
    )

    generator = MultiConditionGenerator(config, seed=seed)

    print(f"Generating {n_samples_per_mechanism} samples per mechanism...")
    all_samples = generator.generate_batch(n_samples_per_mechanism)

    # Shuffle and split
    rng = np.random.default_rng(seed)
    rng.shuffle(all_samples)

    n_val = int(len(all_samples) * val_fraction)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    print(f"Generated {len(train_samples)} train, {len(val_samples)} val samples")

    return train_samples, val_samples
