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

        return {
            'trajectories': trajectories_tensor,  # (max_conditions, n_timepoints, 5)
            'conditions': conditions_tensor,  # (max_conditions, n_condition_features)
            'derived_features': derived_tensor,  # (max_conditions, n_derived_features)
            'condition_mask': condition_mask_tensor,  # (max_conditions,)
            'mechanism_idx': torch.tensor(sample.mechanism_idx, dtype=torch.long),
            'mechanism': sample.mechanism,
            'n_conditions': n_actual,
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

    return {
        'trajectories': trajectories,  # (batch, max_conditions, n_timepoints, 5)
        'conditions': conditions,  # (batch, max_conditions, n_condition_features)
        'derived_features': derived_features,  # (batch, max_conditions, n_derived_features)
        'condition_mask': condition_masks,  # (batch, max_conditions)
        'mechanism_idx': mechanism_idx,  # (batch,)
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
