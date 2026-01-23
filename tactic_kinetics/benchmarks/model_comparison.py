"""
Benchmark 3-5: Model Comparison and Ablation Studies.

This module provides benchmarks for:
3. Model Comparison - Compare TACTIC against baseline methods
4. Ablation Studies - Analyze contribution of different components
5. Transfer Learning - Evaluate generalization to real data

Baselines:
- Random forest on trajectory features
- Simple MLP on trajectories
- TACTIC without thermodynamic priors
- TACTIC without trajectory reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json

from ..mechanisms.templates import get_all_mechanisms
from ..training.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig
from ..models.classifier import MechanismClassifier


@dataclass
class ComparisonMetrics:
    """Model comparison benchmark results."""
    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    n_parameters: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "n_parameters": self.n_parameters,
        }


@dataclass
class AblationMetrics:
    """Ablation study results."""
    component_removed: str
    accuracy_drop: float
    f1_drop: float
    baseline_accuracy: float
    ablated_accuracy: float

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "component_removed": self.component_removed,
            "accuracy_drop": self.accuracy_drop,
            "f1_drop": self.f1_drop,
            "baseline_accuracy": self.baseline_accuracy,
            "ablated_accuracy": self.ablated_accuracy,
        }


class RandomForestBaseline:
    """Random Forest baseline for mechanism classification."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def extract_features(self, trajectories: torch.Tensor) -> np.ndarray:
        """Extract features from trajectories."""
        traj_np = trajectories.numpy()

        features = []
        for traj in traj_np:
            # Basic statistics
            feat = [
                traj.mean(),
                traj.std(),
                traj.min(),
                traj.max(),
                traj[-1] - traj[0],  # Total change
            ]

            # Time derivatives (simple diff)
            diff = np.diff(traj)
            feat.extend([
                diff.mean(),
                diff.std(),
                diff.max(),
            ])

            # Curvature approximation
            diff2 = np.diff(diff)
            feat.extend([
                diff2.mean(),
                diff2.std(),
            ])

            features.append(feat)

        return np.array(features)

    def fit(self, trajectories: torch.Tensor, labels: torch.Tensor):
        """Train the random forest."""
        X = self.extract_features(trajectories)
        y = labels.numpy()
        self.model.fit(X, y)

    def predict(self, trajectories: torch.Tensor) -> np.ndarray:
        """Predict mechanism labels."""
        X = self.extract_features(trajectories)
        return self.model.predict(X)


class MLPBaseline(nn.Module):
    """Simple MLP baseline for mechanism classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        n_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.features(x)
        return self.classifier(h)


@dataclass
class ModelComparisonConfig:
    """Configuration for model comparison benchmark."""
    n_samples_per_mechanism: int = 500
    test_split: float = 0.2
    noise_std: float = 0.01
    n_timepoints: int = 50
    random_seed: int = 42


class ModelComparisonBenchmark:
    """
    Benchmark for comparing different models on mechanism classification.

    Compares:
    1. Random Forest on trajectory features
    2. Simple MLP on trajectory data
    3. TACTIC full model
    4. Various ablations
    """

    def __init__(self, config: Optional[ModelComparisonConfig] = None):
        """Initialize benchmark."""
        self.config = config or ModelComparisonConfig()
        self.mechanisms = get_all_mechanisms()
        self.mechanism_names = list(self.mechanisms.keys())
        self.n_mechanisms = len(self.mechanism_names)

        # Data generator
        self.data_config = SyntheticDataConfig(
            noise_std=self.config.noise_std,
            n_timepoints=self.config.n_timepoints,
            n_samples_per_mechanism=self.config.n_samples_per_mechanism,
        )
        self.data_generator = SyntheticDataGenerator(self.data_config)
        self.batch_size = 32

    def generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate benchmark dataset."""
        all_trajectories = []
        all_labels = []

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        for mech_idx, mech_name in enumerate(self.mechanism_names):
            n_generated = 0
            target = self.config.n_samples_per_mechanism

            while n_generated < target:
                try:
                    sample = self.data_generator.generate_sample(mech_name)

                    if "P" in sample["trajectories"]:
                        traj = sample["trajectories"]["P"]
                    else:
                        product_keys = [k for k in sample["trajectories"].keys() if k in ["P", "Q"]]
                        if product_keys:
                            traj = sample["trajectories"][product_keys[0]]
                        else:
                            continue

                    all_trajectories.append(traj)
                    all_labels.extend([mech_idx] * traj.shape[0])
                    n_generated += traj.shape[0]

                except Exception:
                    continue

        trajectories = torch.cat(all_trajectories, dim=0)
        labels = torch.tensor(all_labels[:trajectories.shape[0]], dtype=torch.long)

        return trajectories, labels

    def train_test_split(
        self,
        trajectories: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split data into train and test sets."""
        n_samples = trajectories.shape[0]
        n_test = int(n_samples * self.config.test_split)

        # Shuffle
        indices = torch.randperm(n_samples)
        trajectories = trajectories[indices]
        labels = labels[indices]

        X_train = trajectories[n_test:]
        y_train = labels[n_test:]
        X_test = trajectories[:n_test]
        y_test = labels[:n_test]

        return X_train, y_train, X_test, y_test

    def evaluate_random_forest(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> ComparisonMetrics:
        """Train and evaluate Random Forest baseline."""
        import time

        model = RandomForestBaseline()

        # Train
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate
        start = time.time()
        y_pred = model.predict(X_test)
        infer_time = time.time() - start

        accuracy = accuracy_score(y_test.numpy(), y_pred)
        macro_f1 = f1_score(y_test.numpy(), y_pred, average='macro')
        weighted_f1 = f1_score(y_test.numpy(), y_pred, average='weighted')

        return ComparisonMetrics(
            model_name="RandomForest",
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            training_time=train_time,
            inference_time=infer_time,
        )

    def evaluate_mlp_baseline(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        n_epochs: int = 50,
    ) -> ComparisonMetrics:
        """Train and evaluate MLP baseline."""
        import time

        model = MLPBaseline(
            input_dim=self.config.n_timepoints,
            n_classes=self.n_mechanisms,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train
        start = time.time()
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        train_time = time.time() - start

        # Evaluate
        model.eval()
        start = time.time()
        with torch.no_grad():
            logits = model(X_test)
            y_pred = logits.argmax(dim=-1)
        infer_time = time.time() - start

        accuracy = (y_pred == y_test).float().mean().item()
        y_pred_np = y_pred.numpy()
        y_test_np = y_test.numpy()
        macro_f1 = f1_score(y_test_np, y_pred_np, average='macro')
        weighted_f1 = f1_score(y_test_np, y_pred_np, average='weighted')

        n_params = sum(p.numel() for p in model.parameters())

        return ComparisonMetrics(
            model_name="MLP",
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            training_time=train_time,
            inference_time=infer_time,
            n_parameters=n_params,
        )

    def evaluate_model(
        self,
        model: nn.Module,
        model_name: str,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        encoder: Optional[nn.Module] = None,
    ) -> ComparisonMetrics:
        """Evaluate a pre-trained model."""
        import time

        model.eval()
        start = time.time()

        with torch.no_grad():
            if encoder is not None:
                features = encoder(X_test)
            else:
                features = X_test

            logits = model(features)
            y_pred = logits.argmax(dim=-1)

        infer_time = time.time() - start

        accuracy = (y_pred == y_test).float().mean().item()
        y_pred_np = y_pred.numpy()
        y_test_np = y_test.numpy()
        macro_f1 = f1_score(y_test_np, y_pred_np, average='macro')
        weighted_f1 = f1_score(y_test_np, y_pred_np, average='weighted')

        n_params = sum(p.numel() for p in model.parameters())
        if encoder is not None:
            n_params += sum(p.numel() for p in encoder.parameters())

        return ComparisonMetrics(
            model_name=model_name,
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            inference_time=infer_time,
            n_parameters=n_params,
        )

    def run_comparison(
        self,
        models: Dict[str, Tuple[nn.Module, Optional[nn.Module]]] = None,
    ) -> Dict[str, ComparisonMetrics]:
        """
        Run full comparison benchmark.

        Args:
            models: Dict mapping model name to (classifier, encoder) tuple
                    The encoder is optional.

        Returns:
            Dict mapping model name to metrics
        """
        print("Generating benchmark dataset...")
        trajectories, labels = self.generate_dataset()
        X_train, y_train, X_test, y_test = self.train_test_split(trajectories, labels)

        print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

        results = {}

        # Baselines
        print("\nEvaluating Random Forest...")
        results["RandomForest"] = self.evaluate_random_forest(X_train, y_train, X_test, y_test)
        print(f"  Accuracy: {results['RandomForest'].accuracy:.3f}")

        print("\nEvaluating MLP Baseline...")
        results["MLP"] = self.evaluate_mlp_baseline(X_train, y_train, X_test, y_test)
        print(f"  Accuracy: {results['MLP'].accuracy:.3f}")

        # Custom models
        if models:
            for name, (classifier, encoder) in models.items():
                print(f"\nEvaluating {name}...")
                results[name] = self.evaluate_model(classifier, name, X_test, y_test, encoder)
                print(f"  Accuracy: {results[name].accuracy:.3f}")

        return results

    def run_ablation(
        self,
        full_model: nn.Module,
        ablated_models: Dict[str, nn.Module],
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        encoder: Optional[nn.Module] = None,
    ) -> List[AblationMetrics]:
        """
        Run ablation study comparing full model to ablated versions.

        Args:
            full_model: The full TACTIC model
            ablated_models: Dict mapping component name to model without that component
            X_test: Test trajectories
            y_test: Test labels
            encoder: Optional encoder

        Returns:
            List of ablation metrics
        """
        # Get baseline performance
        baseline_metrics = self.evaluate_model(full_model, "baseline", X_test, y_test, encoder)
        baseline_accuracy = baseline_metrics.accuracy
        baseline_f1 = baseline_metrics.macro_f1

        results = []
        for component, ablated_model in ablated_models.items():
            ablated_metrics = self.evaluate_model(ablated_model, component, X_test, y_test, encoder)

            results.append(AblationMetrics(
                component_removed=component,
                accuracy_drop=baseline_accuracy - ablated_metrics.accuracy,
                f1_drop=baseline_f1 - ablated_metrics.macro_f1,
                baseline_accuracy=baseline_accuracy,
                ablated_accuracy=ablated_metrics.accuracy,
            ))

        return results

    def comparison_table(self, results: Dict[str, ComparisonMetrics]) -> str:
        """Generate a text table of comparison results."""
        lines = [
            "=" * 80,
            "Model Comparison Results",
            "=" * 80,
            f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'Params':>12}",
            "-" * 80,
        ]

        for name, metrics in sorted(results.items(), key=lambda x: -x[1].accuracy):
            params_str = f"{metrics.n_parameters:,}" if metrics.n_parameters else "N/A"
            lines.append(
                f"{name:<25} {metrics.accuracy:>10.3f} {metrics.macro_f1:>10.3f} "
                f"{metrics.weighted_f1:>12.3f} {params_str:>12}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def save_results(self, results: Dict[str, ComparisonMetrics], path: str):
        """Save benchmark results to JSON."""
        results_dict = {name: m.to_dict() for name, m in results.items()}
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
