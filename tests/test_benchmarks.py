"""Tests for benchmark experiments."""

import pytest
import torch
import numpy as np

from tactic_kinetics.benchmarks.mechanism_classification import (
    MechanismClassificationBenchmark,
    BenchmarkConfig,
    ClassificationMetrics,
)
from tactic_kinetics.benchmarks.energy_reconstruction import (
    EnergyReconstructionBenchmark,
    ReconstructionBenchmarkConfig,
)
from tactic_kinetics.benchmarks.model_comparison import (
    ModelComparisonBenchmark,
    ModelComparisonConfig,
    RandomForestBaseline,
    MLPBaseline,
)
from tactic_kinetics.models.classifier import MechanismClassifier
from tactic_kinetics.mechanisms.templates import get_all_mechanisms


class TestMechanismClassificationBenchmark:
    """Tests for mechanism classification benchmark."""

    def test_benchmark_creation(self):
        """Test creating benchmark."""
        config = BenchmarkConfig(n_samples_per_mechanism=2, n_timepoints=10)
        benchmark = MechanismClassificationBenchmark(config)

        assert benchmark.n_mechanisms == 10
        assert len(benchmark.mechanism_names) == 10

    def test_classification_metrics_creation(self):
        """Test creating classification metrics."""
        metrics = ClassificationMetrics(
            accuracy=0.85,
            per_class_precision={"mm": 0.9, "comp_inhib": 0.8},
            per_class_recall={"mm": 0.85, "comp_inhib": 0.82},
            per_class_f1={"mm": 0.87, "comp_inhib": 0.81},
            confusion_matrix=np.eye(2),
            mechanism_names=["mm", "comp_inhib"],
            n_samples=100,
        )

        assert metrics.accuracy == 0.85
        assert metrics.n_samples == 100

    def test_classification_metrics_summary(self):
        """Test metrics summary generation."""
        metrics = ClassificationMetrics(
            accuracy=0.85,
            per_class_precision={"mm": 0.9, "comp_inhib": 0.8},
            per_class_recall={"mm": 0.85, "comp_inhib": 0.82},
            per_class_f1={"mm": 0.87, "comp_inhib": 0.81},
            confusion_matrix=np.eye(2),
            mechanism_names=["mm", "comp_inhib"],
            n_samples=100,
        )

        summary = metrics.summary()
        assert "Accuracy: 0.850" in summary
        assert "mm" in summary

    def test_classification_metrics_to_dict(self):
        """Test converting metrics to dict."""
        metrics = ClassificationMetrics(
            accuracy=0.85,
            per_class_precision={"mm": 0.9},
            per_class_recall={"mm": 0.85},
            per_class_f1={"mm": 0.87},
            confusion_matrix=np.eye(2),
            mechanism_names=["mm", "comp_inhib"],
            n_samples=100,
        )

        d = metrics.to_dict()
        assert d["accuracy"] == 0.85
        assert d["n_samples"] == 100
        assert "confusion_matrix" in d


class TestEnergyReconstructionBenchmark:
    """Tests for energy reconstruction benchmark."""

    def test_benchmark_creation(self):
        """Test creating benchmark."""
        config = ReconstructionBenchmarkConfig(n_samples=2, n_timepoints=10)
        benchmark = EnergyReconstructionBenchmark(config)

        assert len(benchmark.mechanisms) == 10


class TestModelComparisonBenchmark:
    """Tests for model comparison benchmark."""

    def test_benchmark_creation(self):
        """Test creating benchmark."""
        config = ModelComparisonConfig(n_samples_per_mechanism=2, n_timepoints=10)
        benchmark = ModelComparisonBenchmark(config)

        assert benchmark.n_mechanisms == 10

    def test_random_forest_baseline_init(self):
        """Test Random Forest baseline initialization."""
        rf = RandomForestBaseline()
        assert rf.model is not None

    def test_random_forest_feature_extraction(self):
        """Test trajectory feature extraction."""
        rf = RandomForestBaseline()

        trajectories = torch.randn(10, 50)
        features = rf.extract_features(trajectories)

        # Should extract 10 features per trajectory
        assert features.shape[0] == 10
        assert features.shape[1] == 10

    def test_random_forest_fit_predict(self):
        """Test Random Forest fit and predict."""
        rf = RandomForestBaseline()

        trajectories = torch.randn(100, 50)
        labels = torch.randint(0, 5, (100,))

        rf.fit(trajectories, labels)
        predictions = rf.predict(trajectories)

        assert predictions.shape == (100,)
        assert all(0 <= p < 5 for p in predictions)

    def test_mlp_baseline_init(self):
        """Test MLP baseline initialization."""
        mlp = MLPBaseline(input_dim=50, n_classes=10)
        assert mlp is not None

    def test_mlp_baseline_forward(self):
        """Test MLP baseline forward pass."""
        mlp = MLPBaseline(input_dim=50, n_classes=10)

        x = torch.randn(32, 50)
        logits = mlp(x)

        assert logits.shape == (32, 10)

    def test_mlp_baseline_gradient_flow(self):
        """Test that gradients flow through MLP."""
        mlp = MLPBaseline(input_dim=50, n_classes=10)

        x = torch.randn(16, 50, requires_grad=True)
        logits = mlp(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None


class TestBenchmarkConfigs:
    """Tests for benchmark configurations."""

    def test_benchmark_config_defaults(self):
        """Test BenchmarkConfig defaults."""
        config = BenchmarkConfig()

        assert config.n_samples_per_mechanism == 500
        assert config.test_split == 0.2
        assert len(config.noise_levels) == 3

    def test_reconstruction_config_defaults(self):
        """Test ReconstructionBenchmarkConfig defaults."""
        config = ReconstructionBenchmarkConfig()

        assert config.n_samples == 1000
        assert config.noise_std == 0.01
        assert config.n_timepoints == 50

    def test_comparison_config_defaults(self):
        """Test ModelComparisonConfig defaults."""
        config = ModelComparisonConfig()

        assert config.n_samples_per_mechanism == 500
        assert config.test_split == 0.2
        assert config.noise_std == 0.01
