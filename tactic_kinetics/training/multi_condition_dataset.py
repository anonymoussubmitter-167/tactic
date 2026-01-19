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
    max_conditions: int = 10  # Maximum number of conditions per sample
    n_timepoints: int = 20  # Fixed number of timepoints
    n_condition_features: int = 8  # Number of condition features


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

        for traj_data in sample.trajectories[:self.config.max_conditions]:
            # Get the main concentration trajectory (S or P, depending on mechanism)
            conc = self._get_main_trajectory(traj_data['concentrations'], sample.mechanism)
            t = traj_data['t']

            # Interpolate to fixed number of timepoints if needed
            if len(t) != self.config.n_timepoints:
                t_new = np.linspace(t[0], t[-1], self.config.n_timepoints)
                conc = np.interp(t_new, t, conc)
                t = t_new

            # Compute rate (d_conc/dt)
            rate = np.gradient(conc, t)

            # Normalize
            t_norm = t / self.time_max
            conc_norm = (conc - self.conc_mean) / self.conc_std
            rate_norm = rate / (self.conc_std / self.time_max + 1e-8)

            # Stack features: (n_timepoints, 3)
            traj_features = np.stack([t_norm, conc_norm, rate_norm], axis=-1)
            trajectories.append(traj_features)

            # Extract condition features
            cond_features = self._extract_condition_features(traj_data['conditions'])
            conditions_list.append(cond_features)

        # Pad to max_conditions
        n_actual = len(trajectories)
        n_pad = self.config.max_conditions - n_actual

        if n_pad > 0:
            pad_traj = np.zeros((self.config.n_timepoints, 3))
            pad_cond = np.zeros(self.config.n_condition_features)
            for _ in range(n_pad):
                trajectories.append(pad_traj)
                conditions_list.append(pad_cond)

        # Create mask (True = invalid/padded)
        condition_mask = np.array([False] * n_actual + [True] * n_pad)

        # Convert to tensors
        trajectories_tensor = torch.tensor(np.stack(trajectories), dtype=torch.float32)
        conditions_tensor = torch.tensor(np.stack(conditions_list), dtype=torch.float32)
        condition_mask_tensor = torch.tensor(condition_mask, dtype=torch.bool)

        return {
            'trajectories': trajectories_tensor,  # (max_conditions, n_timepoints, 3)
            'conditions': conditions_tensor,  # (max_conditions, n_condition_features)
            'condition_mask': condition_mask_tensor,  # (max_conditions,)
            'mechanism_idx': torch.tensor(sample.mechanism_idx, dtype=torch.long),
            'mechanism': sample.mechanism,
            'n_conditions': n_actual,
        }

    def _get_main_trajectory(self, concentrations: Dict[str, np.ndarray], mechanism: str) -> np.ndarray:
        """Get the main trajectory to track (usually product or substrate)."""
        # For most mechanisms, track substrate consumption
        if 'S' in concentrations:
            return concentrations['S']
        elif 'A' in concentrations:
            return concentrations['A']
        elif 'P' in concentrations:
            return concentrations['P']
        else:
            # Return first available
            return list(concentrations.values())[0]

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
    condition_masks = torch.stack([item['condition_mask'] for item in batch])
    mechanism_idx = torch.stack([item['mechanism_idx'] for item in batch])

    return {
        'trajectories': trajectories,  # (batch, max_conditions, n_timepoints, 3)
        'conditions': conditions,  # (batch, max_conditions, n_condition_features)
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
        n_conditions_per_sample=5,
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
