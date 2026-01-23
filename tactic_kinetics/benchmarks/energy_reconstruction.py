"""
Benchmark 2: Energy Profile Reconstruction.

This benchmark evaluates how well the model can recover the underlying
Gibbs energy landscape from kinetics trajectory data.

Metrics:
- State energy MAE/RMSE
- Barrier energy MAE/RMSE
- Rate constant reconstruction error
- Correlation between true and predicted energies
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
import json

from ..mechanisms.templates import get_all_mechanisms, get_mechanism_by_name
from ..training.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig
from ..models.ode_simulator import EnergyToRateConverter


@dataclass
class EnergyReconstructionMetrics:
    """Energy reconstruction benchmark results."""
    state_energy_mae: float
    state_energy_rmse: float
    state_energy_correlation: float
    barrier_energy_mae: float
    barrier_energy_rmse: float
    barrier_energy_correlation: float
    rate_mae: float
    rate_rmse: float
    n_samples: int
    mechanism_name: str

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "state_energy_mae": self.state_energy_mae,
            "state_energy_rmse": self.state_energy_rmse,
            "state_energy_correlation": self.state_energy_correlation,
            "barrier_energy_mae": self.barrier_energy_mae,
            "barrier_energy_rmse": self.barrier_energy_rmse,
            "barrier_energy_correlation": self.barrier_energy_correlation,
            "rate_mae": self.rate_mae,
            "rate_rmse": self.rate_rmse,
            "n_samples": self.n_samples,
            "mechanism_name": self.mechanism_name,
        }

    def summary(self) -> str:
        """Get text summary of results."""
        lines = [
            "=" * 60,
            f"Energy Reconstruction Benchmark: {self.mechanism_name}",
            "=" * 60,
            f"Samples evaluated: {self.n_samples}",
            "",
            "State Energies:",
            f"  MAE: {self.state_energy_mae:.3f} kJ/mol",
            f"  RMSE: {self.state_energy_rmse:.3f} kJ/mol",
            f"  Correlation: {self.state_energy_correlation:.3f}",
            "",
            "Barrier Energies:",
            f"  MAE: {self.barrier_energy_mae:.3f} kJ/mol",
            f"  RMSE: {self.barrier_energy_rmse:.3f} kJ/mol",
            f"  Correlation: {self.barrier_energy_correlation:.3f}",
            "",
            "Rate Constants:",
            f"  MAE: {self.rate_mae:.3f}",
            f"  RMSE: {self.rate_rmse:.3f}",
        ]
        return "\n".join(lines)


@dataclass
class ReconstructionBenchmarkConfig:
    """Configuration for energy reconstruction benchmark."""
    n_samples: int = 1000
    noise_std: float = 0.01
    time_range: Tuple[float, float] = (0.0, 100.0)
    n_timepoints: int = 50
    random_seed: int = 42

    # Energy parameter ranges for ground truth
    state_energy_range: Tuple[float, float] = (-20.0, 20.0)
    barrier_energy_range: Tuple[float, float] = (50.0, 80.0)


class EnergyReconstructionBenchmark:
    """
    Benchmark for evaluating energy profile reconstruction accuracy.

    Tests how well the model can recover the true Gibbs energy landscape
    from noisy kinetics trajectory data.
    """

    def __init__(self, config: Optional[ReconstructionBenchmarkConfig] = None):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config or ReconstructionBenchmarkConfig()
        self.mechanisms = get_all_mechanisms()

        # Setup data generator
        self.data_config = SyntheticDataConfig(
            noise_std=self.config.noise_std,
            n_timepoints=self.config.n_timepoints,
            t_max=self.config.time_range[1],
        )
        self.data_generator = SyntheticDataGenerator(self.data_config)
        self.batch_size = 32

    def generate_dataset(
        self,
        mechanism_name: str,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate benchmark dataset with known ground truth energies.

        Args:
            mechanism_name: Name of mechanism to test
            n_samples: Number of samples to generate

        Returns:
            Dict with keys:
                "trajectories": Product concentration trajectories
                "true_state_energies": Ground truth state energies
                "true_barrier_energies": Ground truth barrier energies
                "true_forward_rates": Ground truth forward rates
                "true_reverse_rates": Ground truth reverse rates
                "time_points": Time values
        """
        n_samples = n_samples or self.config.n_samples
        mech = self.mechanisms[mechanism_name]

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        all_trajectories = []
        all_state_energies = []
        all_barrier_energies = []
        all_forward_rates = []
        all_reverse_rates = []

        converter = EnergyToRateConverter(mech)

        for _ in range(0, n_samples, self.batch_size):
            batch_size = min(self.batch_size, n_samples - len(all_state_energies) * self.batch_size)
            if batch_size <= 0:
                break

            try:
                sample = self.data_generator.generate_sample(mechanism_name)

                # Extract trajectory
                if "P" in sample["trajectories"]:
                    traj = sample["trajectories"]["P"]
                else:
                    product_keys = [k for k in sample["trajectories"].keys() if k in ["P", "Q"]]
                    if product_keys:
                        traj = sample["trajectories"][product_keys[0]]
                    else:
                        continue

                # Get ground truth energies
                state_e = sample["state_energies"]
                barrier_e = sample["barrier_energies"]

                # Compute rates from energies
                forward_rates, reverse_rates = converter(state_e, barrier_e)

                all_trajectories.append(traj)
                all_state_energies.append(state_e)
                all_barrier_energies.append(barrier_e)
                all_forward_rates.append(forward_rates)
                all_reverse_rates.append(reverse_rates)

            except Exception as e:
                print(f"Warning: Failed to generate sample: {e}")
                continue

        if not all_trajectories:
            raise RuntimeError(f"Failed to generate trajectory data for {mechanism_name}")

        t_eval = torch.linspace(
            self.config.time_range[0],
            self.config.time_range[1],
            self.config.n_timepoints,
        )

        return {
            "trajectories": torch.cat(all_trajectories, dim=0),
            "true_state_energies": torch.cat(all_state_energies, dim=0),
            "true_barrier_energies": torch.cat(all_barrier_energies, dim=0),
            "true_forward_rates": torch.cat(all_forward_rates, dim=0),
            "true_reverse_rates": torch.cat(all_reverse_rates, dim=0),
            "time_points": t_eval,
        }

    def evaluate_model(
        self,
        model: torch.nn.Module,
        data: Dict[str, torch.Tensor],
        mechanism_name: str,
    ) -> EnergyReconstructionMetrics:
        """
        Evaluate a model's energy reconstruction accuracy.

        Args:
            model: Model that takes trajectories and outputs energy predictions
            data: Dataset from generate_dataset()
            mechanism_name: Name of mechanism being tested

        Returns:
            Energy reconstruction metrics
        """
        mech = self.mechanisms[mechanism_name]
        model.eval()

        with torch.no_grad():
            # Get model predictions
            # Assumes model returns dict with "state_energies" and "barrier_energies"
            outputs = model(data["trajectories"])

            pred_state_e = outputs.get("state_energies", outputs.get("state_energy"))
            pred_barrier_e = outputs.get("barrier_energies", outputs.get("barrier_energy"))

        true_state_e = data["true_state_energies"]
        true_barrier_e = data["true_barrier_energies"]

        # State energy metrics
        state_diff = pred_state_e - true_state_e
        state_mae = state_diff.abs().mean().item()
        state_rmse = (state_diff ** 2).mean().sqrt().item()

        # Correlation
        pred_flat = pred_state_e.flatten().numpy()
        true_flat = true_state_e.flatten().numpy()
        state_corr, _ = stats.pearsonr(pred_flat, true_flat)

        # Barrier energy metrics
        barrier_diff = pred_barrier_e - true_barrier_e
        barrier_mae = barrier_diff.abs().mean().item()
        barrier_rmse = (barrier_diff ** 2).mean().sqrt().item()

        pred_barrier_flat = pred_barrier_e.flatten().numpy()
        true_barrier_flat = true_barrier_e.flatten().numpy()
        barrier_corr, _ = stats.pearsonr(pred_barrier_flat, true_barrier_flat)

        # Rate reconstruction
        converter = EnergyToRateConverter(mech)
        pred_forward, pred_reverse = converter(pred_state_e, pred_barrier_e)
        true_forward = data["true_forward_rates"]
        true_reverse = data["true_reverse_rates"]

        # Use log-scale for rate comparison
        rate_diff = torch.log10(pred_forward + 1e-10) - torch.log10(true_forward + 1e-10)
        rate_mae = rate_diff.abs().mean().item()
        rate_rmse = (rate_diff ** 2).mean().sqrt().item()

        return EnergyReconstructionMetrics(
            state_energy_mae=state_mae,
            state_energy_rmse=state_rmse,
            state_energy_correlation=state_corr,
            barrier_energy_mae=barrier_mae,
            barrier_energy_rmse=barrier_rmse,
            barrier_energy_correlation=barrier_corr,
            rate_mae=rate_mae,
            rate_rmse=rate_rmse,
            n_samples=data["trajectories"].shape[0],
            mechanism_name=mechanism_name,
        )

    def run_all_mechanisms(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, EnergyReconstructionMetrics]:
        """
        Evaluate reconstruction across all mechanisms.

        Args:
            model: The model to evaluate

        Returns:
            Dict mapping mechanism name to metrics
        """
        results = {}

        for mech_name in self.mechanisms.keys():
            print(f"Evaluating {mech_name}...")
            try:
                data = self.generate_dataset(mech_name)
                metrics = self.evaluate_model(model, data, mech_name)
                results[mech_name] = metrics
                print(f"  State Energy MAE: {metrics.state_energy_mae:.3f} kJ/mol")
                print(f"  Barrier Energy MAE: {metrics.barrier_energy_mae:.3f} kJ/mol")
            except Exception as e:
                print(f"  Failed: {e}")

        return results

    def run_noise_sensitivity(
        self,
        model: torch.nn.Module,
        mechanism_name: str,
        noise_levels: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1],
    ) -> Dict[float, EnergyReconstructionMetrics]:
        """
        Evaluate reconstruction accuracy at different noise levels.

        Args:
            model: The model to evaluate
            mechanism_name: Mechanism to test
            noise_levels: List of noise standard deviations

        Returns:
            Dict mapping noise level to metrics
        """
        results = {}

        for noise in noise_levels:
            print(f"Noise level: {noise}")
            self.data_config.noise_std = noise
            data = self.generate_dataset(mechanism_name)
            metrics = self.evaluate_model(model, data, mechanism_name)
            results[noise] = metrics
            print(f"  State Energy MAE: {metrics.state_energy_mae:.3f} kJ/mol")

        return results

    def save_results(
        self,
        results: Dict[str, EnergyReconstructionMetrics],
        path: str,
    ):
        """Save benchmark results to JSON."""
        results_dict = {name: m.to_dict() for name, m in results.items()}
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
