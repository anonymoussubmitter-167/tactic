"""
Benchmark 1: Mechanism Classification from Trajectory Data.

This benchmark evaluates how well the model can identify the underlying
enzyme mechanism from kinetics trajectory data.

Metrics:
- Overall accuracy
- Per-mechanism precision/recall/F1
- Confusion matrix
- Confidence calibration
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..mechanisms.templates import get_all_mechanisms, get_mechanism_by_name
from ..training.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig
from ..models.classifier import MechanismClassifier


@dataclass
class ClassificationMetrics:
    """Classification benchmark results."""
    accuracy: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    mechanism_names: List[str]
    n_samples: int

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "accuracy": self.accuracy,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "mechanism_names": self.mechanism_names,
            "n_samples": self.n_samples,
        }

    def summary(self) -> str:
        """Get text summary of results."""
        lines = [
            "=" * 60,
            "Mechanism Classification Benchmark Results",
            "=" * 60,
            f"Overall Accuracy: {self.accuracy:.3f}",
            f"Total Samples: {self.n_samples}",
            "",
            "Per-Class F1 Scores:",
        ]
        for name in self.mechanism_names:
            f1 = self.per_class_f1.get(name, 0)
            prec = self.per_class_precision.get(name, 0)
            rec = self.per_class_recall.get(name, 0)
            lines.append(f"  {name}: F1={f1:.3f} (P={prec:.3f}, R={rec:.3f})")

        return "\n".join(lines)


@dataclass
class BenchmarkConfig:
    """Configuration for mechanism classification benchmark."""
    n_samples_per_mechanism: int = 500
    test_split: float = 0.2
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    time_range: Tuple[float, float] = (0.0, 100.0)
    n_timepoints: int = 50
    random_seed: int = 42

    # Energy parameter ranges
    state_energy_range: Tuple[float, float] = (-20.0, 20.0)
    barrier_energy_range: Tuple[float, float] = (50.0, 80.0)


class MechanismClassificationBenchmark:
    """
    Benchmark for evaluating mechanism classification accuracy.

    Tests the model's ability to correctly identify the enzyme mechanism
    from trajectory data under various conditions.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.mechanisms = get_all_mechanisms()
        self.mechanism_names = list(self.mechanisms.keys())
        self.n_mechanisms = len(self.mechanism_names)

        # Setup data generator
        self.data_config = SyntheticDataConfig(
            noise_std=self.config.noise_levels[0],
            n_timepoints=self.config.n_timepoints,
            t_max=self.config.time_range[1],
            n_samples_per_mechanism=self.config.n_samples_per_mechanism,
        )
        self.data_generator = SyntheticDataGenerator(self.data_config)
        self.batch_size = 32

    def generate_dataset(
        self,
        n_samples_per_mechanism: Optional[int] = None,
        noise_std: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate benchmark dataset.

        Args:
            n_samples_per_mechanism: Number of samples per mechanism
            noise_std: Noise level for trajectories

        Returns:
            Tuple of (trajectories, labels, time_points):
                trajectories: (n_samples, n_timepoints) product concentration
                labels: (n_samples,) mechanism indices
                time_points: (n_timepoints,) time values
        """
        n_per_mech = n_samples_per_mechanism or self.config.n_samples_per_mechanism

        # Update noise level
        self.data_config.noise_std = noise_std

        all_trajectories = []
        all_labels = []

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        for mech_idx, mech_name in enumerate(self.mechanism_names):
            mech = self.mechanisms[mech_name]

            for _ in range(0, n_per_mech, self.batch_size):
                batch_size = min(self.batch_size, n_per_mech - len([l for l in all_labels if l == mech_idx]))
                if batch_size <= 0:
                    break

                try:
                    sample = self.data_generator.generate_sample(mech_name)

                    # Extract product trajectory
                    if "P" in sample["trajectories"]:
                        traj = sample["trajectories"]["P"]
                    else:
                        # For bi-substrate mechanisms, use first product
                        product_keys = [k for k in sample["trajectories"].keys() if k in ["P", "Q"]]
                        if product_keys:
                            traj = sample["trajectories"][product_keys[0]]
                        else:
                            continue

                    all_trajectories.append(traj)
                    all_labels.extend([mech_idx] * traj.shape[0])
                except Exception as e:
                    print(f"Warning: Failed to generate sample for {mech_name}: {e}")
                    continue

        if not all_trajectories:
            raise RuntimeError("Failed to generate any trajectory data")

        trajectories = torch.cat(all_trajectories, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)

        # Generate time points
        t_eval = torch.linspace(
            self.config.time_range[0],
            self.config.time_range[1],
            self.config.n_timepoints,
        )

        return trajectories, labels, t_eval

    def evaluate_model(
        self,
        model: MechanismClassifier,
        trajectories: torch.Tensor,
        labels: torch.Tensor,
        encoder: Optional[torch.nn.Module] = None,
    ) -> ClassificationMetrics:
        """
        Evaluate a classifier on the benchmark dataset.

        Args:
            model: The classifier to evaluate
            trajectories: Trajectory data (n_samples, n_timepoints)
            labels: Ground truth labels (n_samples,)
            encoder: Optional encoder to convert trajectories to latent space

        Returns:
            Classification metrics
        """
        model.eval()

        with torch.no_grad():
            if encoder is not None:
                # Encode trajectories to latent space
                latent = encoder(trajectories)
            else:
                # Assume trajectories are already features
                latent = trajectories

            # Get predictions
            logits = model(latent)
            predictions = logits.argmax(dim=-1)

        # Compute metrics
        n_samples = labels.shape[0]
        correct = (predictions == labels).sum().item()
        accuracy = correct / n_samples

        # Per-class metrics
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}

        confusion = torch.zeros(self.n_mechanisms, self.n_mechanisms)

        for true_label, pred_label in zip(labels.tolist(), predictions.tolist()):
            confusion[true_label, pred_label] += 1

        for i, mech_name in enumerate(self.mechanism_names):
            # True positives, false positives, false negatives
            tp = confusion[i, i].item()
            fp = confusion[:, i].sum().item() - tp
            fn = confusion[i, :].sum().item() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_precision[mech_name] = precision
            per_class_recall[mech_name] = recall
            per_class_f1[mech_name] = f1

        return ClassificationMetrics(
            accuracy=accuracy,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            confusion_matrix=confusion.numpy(),
            mechanism_names=self.mechanism_names,
            n_samples=n_samples,
        )

    def run_noise_ablation(
        self,
        model: MechanismClassifier,
        encoder: Optional[torch.nn.Module] = None,
        noise_levels: Optional[List[float]] = None,
    ) -> Dict[float, ClassificationMetrics]:
        """
        Run classification benchmark at different noise levels.

        Args:
            model: The classifier to evaluate
            encoder: Optional trajectory encoder
            noise_levels: List of noise levels to test

        Returns:
            Dict mapping noise level to metrics
        """
        noise_levels = noise_levels or self.config.noise_levels
        results = {}

        for noise in noise_levels:
            print(f"Evaluating at noise level: {noise}")
            trajectories, labels, _ = self.generate_dataset(noise_std=noise)
            metrics = self.evaluate_model(model, trajectories, labels, encoder)
            results[noise] = metrics
            print(f"  Accuracy: {metrics.accuracy:.3f}")

        return results

    def run_sample_efficiency(
        self,
        model: MechanismClassifier,
        encoder: Optional[torch.nn.Module] = None,
        sample_sizes: List[int] = [50, 100, 200, 500, 1000],
    ) -> Dict[int, ClassificationMetrics]:
        """
        Evaluate how classification accuracy varies with dataset size.

        Args:
            model: The classifier to evaluate
            encoder: Optional trajectory encoder
            sample_sizes: List of per-mechanism sample sizes to test

        Returns:
            Dict mapping sample size to metrics
        """
        results = {}

        for n_samples in sample_sizes:
            print(f"Evaluating with {n_samples} samples per mechanism")
            trajectories, labels, _ = self.generate_dataset(n_samples_per_mechanism=n_samples)
            metrics = self.evaluate_model(model, trajectories, labels, encoder)
            results[n_samples] = metrics
            print(f"  Accuracy: {metrics.accuracy:.3f}")

        return results

    def save_results(self, results: ClassificationMetrics, path: str):
        """Save benchmark results to JSON."""
        with open(path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
