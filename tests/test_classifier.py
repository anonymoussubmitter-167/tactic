"""Tests for mechanism classifier."""

import pytest
import torch
import numpy as np

from tactic_kinetics.models.classifier import (
    MechanismClassifier,
    HierarchicalClassifier,
)
from tactic_kinetics.mechanisms.templates import get_all_mechanisms, get_mechanism_by_name


class TestMechanismClassifier:
    """Tests for MechanismClassifier."""

    def test_classifier_creation(self):
        """Test creating MechanismClassifier."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=64,
            mechanism_names=mechanisms,
        )

        assert len(classifier.mechanism_names) == 10
        assert classifier.n_mechanisms == 10

    def test_classifier_forward(self):
        """Test classifier forward pass."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=64,
            mechanism_names=mechanisms,
        )

        batch_size = 8
        x = torch.randn(batch_size, 64)

        logits = classifier(x)

        assert logits.shape == (batch_size, 10)

    def test_classifier_predict_probs(self):
        """Test getting mechanism probabilities."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=64,
            mechanism_names=mechanisms,
        )

        x = torch.randn(4, 64)
        probs = classifier.predict_probs(x)

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_classifier_predict_mechanism(self):
        """Test mechanism prediction."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=64,
            mechanism_names=mechanisms,
        )

        x = torch.randn(4, 64)
        predictions, confidences = classifier.predict_mechanism(x)

        assert len(predictions) == 4
        assert confidences.shape == (4,)
        # Each prediction should be a valid mechanism name
        for pred in predictions:
            assert pred in mechanisms
        # Confidences should be between 0 and 1
        assert (confidences >= 0).all()
        assert (confidences <= 1).all()

    def test_classifier_compute_loss(self):
        """Test loss computation."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=64,
            mechanism_names=mechanisms,
        )

        x = torch.randn(8, 64)
        labels = torch.randint(0, 10, (8,))

        loss = classifier.compute_loss(x, labels)

        assert loss.shape == ()
        assert loss >= 0

    def test_classifier_gradient_flow(self):
        """Test that gradients flow through classifier."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=64,
            mechanism_names=mechanisms,
        )

        x = torch.randn(4, 64, requires_grad=True)
        logits = classifier(x)

        # Compute loss and backprop
        loss = logits.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestHierarchicalClassifier:
    """Tests for HierarchicalClassifier."""

    def test_hierarchical_classifier_creation(self):
        """Test creating hierarchical classifier."""
        hierarchy = {
            "standard": ["michaelis_menten_irreversible", "michaelis_menten_reversible"],
            "inhibition": ["competitive_inhibition", "uncompetitive_inhibition", "mixed_inhibition"],
            "multi_substrate": ["ordered_bi_bi", "random_bi_bi", "ping_pong"],
        }

        classifier = HierarchicalClassifier(
            d_input=64,
            mechanism_hierarchy=hierarchy,
        )

        assert classifier.n_categories == 3
        assert len(classifier.all_mechanisms) == 8

    def test_hierarchical_classifier_forward(self):
        """Test hierarchical classifier forward pass."""
        hierarchy = {
            "standard": ["michaelis_menten_irreversible", "michaelis_menten_reversible"],
            "inhibition": ["competitive_inhibition", "uncompetitive_inhibition", "mixed_inhibition"],
        }

        classifier = HierarchicalClassifier(
            d_input=64,
            mechanism_hierarchy=hierarchy,
        )

        x = torch.randn(4, 64)
        output = classifier(x)

        assert "category_logits" in output
        assert "mechanism_logits" in output
        assert output["category_logits"].shape == (4, 2)
        assert output["mechanism_logits"].shape == (4, 5)

    def test_hierarchical_classifier_predict(self):
        """Test hierarchical prediction."""
        hierarchy = {
            "standard": ["michaelis_menten_irreversible"],
            "inhibition": ["competitive_inhibition", "uncompetitive_inhibition"],
        }

        classifier = HierarchicalClassifier(
            d_input=32,
            mechanism_hierarchy=hierarchy,
        )

        x = torch.randn(4, 32)
        predictions, confidences = classifier.predict_mechanism(x)

        assert len(predictions) == 4
        assert confidences.shape == (4,)
        # Predictions should be in mechanism list
        for pred in predictions:
            assert pred in classifier.all_mechanisms


class TestClassifierIntegration:
    """Integration tests for classifier with energy profiles."""

    def test_classify_from_latent_features(self):
        """Test classifying from latent features."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=32,
            mechanism_names=mechanisms,
        )

        # Simulate latent features
        batch_size = 8
        latent_features = torch.randn(batch_size, 32)

        # Get predictions
        logits = classifier(latent_features)
        probs = torch.softmax(logits, dim=-1)

        assert probs.shape == (batch_size, 10)

    def test_classifier_training_step(self):
        """Test a single training step."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=32,
            mechanism_names=mechanisms,
        )

        # Create fake batch
        batch_size = 16
        x = torch.randn(batch_size, 32)
        labels = torch.randint(0, 10, (batch_size,))

        # Forward pass
        loss = classifier.compute_loss(x, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in classifier.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestClassifierMetrics:
    """Tests for classifier evaluation metrics."""

    def test_accuracy_computation(self):
        """Test computing classification accuracy."""
        mechanisms = list(get_all_mechanisms().keys())
        classifier = MechanismClassifier(
            d_input=32,
            mechanism_names=mechanisms,
        )

        # Create test data
        n_samples = 100
        x = torch.randn(n_samples, 32)
        true_labels = torch.randint(0, 10, (n_samples,))

        # Get predictions
        logits = classifier(x)
        pred_labels = logits.argmax(dim=-1)

        # Compute accuracy
        accuracy = (pred_labels == true_labels).float().mean()

        # We just check it's a valid number between 0 and 1
        assert 0 <= accuracy <= 1

    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        mechanisms = list(get_all_mechanisms().keys())
        n_classes = len(mechanisms)

        # Simulate predictions and labels
        n_samples = 100
        pred_labels = torch.randint(0, n_classes, (n_samples,))
        true_labels = torch.randint(0, n_classes, (n_samples,))

        # Build confusion matrix
        confusion = torch.zeros(n_classes, n_classes)
        for pred, true in zip(pred_labels, true_labels):
            confusion[true, pred] += 1

        assert confusion.shape == (n_classes, n_classes)
        assert confusion.sum() == n_samples
