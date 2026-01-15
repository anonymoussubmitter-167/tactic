"""
Mechanism classifier for TACTIC-Kinetics.

This module implements a classifier that predicts mechanism probabilities
from the latent representation, enabling mechanism discrimination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from ..mechanisms.base import MechanismTemplate


class MechanismClassifier(nn.Module):
    """
    Classifier for enzyme mechanism discrimination.

    Takes a latent representation and outputs probabilities over different
    mechanism types.
    """

    def __init__(
        self,
        d_input: int,
        mechanism_names: List[str],
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        """
        Args:
            d_input: Dimension of input latent representation
            mechanism_names: List of mechanism names to classify
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        self.mechanism_names = mechanism_names
        self.n_mechanisms = len(mechanism_names)
        self.mechanism_to_idx = {name: i for i, name in enumerate(mechanism_names)}

        # Build MLP classifier
        layers = []
        in_dim = d_input
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, self.n_mechanisms)

    def forward(self, h: Tensor) -> Tensor:
        """
        Predict mechanism probabilities.

        Args:
            h: Latent representation, shape (batch, d_input)

        Returns:
            Mechanism logits, shape (batch, n_mechanisms)
        """
        features = self.mlp(h)
        logits = self.classifier(features)
        return logits

    def predict_probs(self, h: Tensor) -> Tensor:
        """
        Get mechanism probabilities (softmax of logits).

        Args:
            h: Latent representation, shape (batch, d_input)

        Returns:
            Probabilities, shape (batch, n_mechanisms)
        """
        logits = self.forward(h)
        return F.softmax(logits, dim=-1)

    def predict_mechanism(self, h: Tensor) -> Tuple[List[str], Tensor]:
        """
        Predict the most likely mechanism for each sample.

        Args:
            h: Latent representation, shape (batch, d_input)

        Returns:
            Tuple of (mechanism_names, confidences):
                mechanism_names: List of predicted mechanism names
                confidences: Tensor of confidence scores, shape (batch,)
        """
        probs = self.predict_probs(h)
        confidences, indices = probs.max(dim=-1)

        predicted_names = [self.mechanism_names[i] for i in indices.tolist()]
        return predicted_names, confidences

    def compute_loss(
        self,
        h: Tensor,
        labels: Tensor,
        label_smoothing: float = 0.1,
    ) -> Tensor:
        """
        Compute cross-entropy loss for mechanism classification.

        Args:
            h: Latent representation, shape (batch, d_input)
            labels: Ground truth mechanism indices, shape (batch,)
            label_smoothing: Label smoothing factor

        Returns:
            Loss scalar
        """
        logits = self.forward(h)
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


class EnergyProfileClassifier(nn.Module):
    """
    Classifier that operates on energy profiles rather than raw latent features.

    This classifier first predicts energy landscapes for each mechanism,
    then classifies based on the energy profile shapes.
    """

    def __init__(
        self,
        mechanisms: Dict[str, MechanismTemplate],
        d_input: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ):
        """
        Args:
            mechanisms: Dict mapping mechanism names to templates
            d_input: Dimension of input (concatenated energy features)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        self.mechanism_names = list(mechanisms.keys())
        self.mechanisms = mechanisms
        self.n_mechanisms = len(mechanisms)

        # Compute expected input dimension
        # For each mechanism: n_state_energies + n_barrier_energies + derived features
        self.input_dims = {}
        total_dim = 0
        for name, mech in mechanisms.items():
            mech_dim = mech.n_energy_params + mech.n_barrier_params + 3  # +3 for derived features
            self.input_dims[name] = mech_dim
            total_dim += mech_dim

        # MLP classifier on concatenated energy features
        layers = []
        in_dim = total_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, self.n_mechanisms)

    def extract_energy_features(
        self,
        state_energies: Tensor,
        barrier_energies: Tensor,
        mechanism: MechanismTemplate,
    ) -> Tensor:
        """
        Extract features from energy profile.

        Args:
            state_energies: shape (batch, n_states)
            barrier_energies: shape (batch, n_barriers)
            mechanism: The mechanism template

        Returns:
            Features, shape (batch, n_features)
        """
        # Basic features: energies themselves
        basic = torch.cat([state_energies, barrier_energies], dim=-1)

        # Derived features
        # 1. Overall reaction energy (last state - first state equivalent)
        dg_rxn = state_energies[:, -1] - state_energies[:, 0] if state_energies.shape[1] > 1 else state_energies[:, 0]

        # 2. Maximum barrier height
        max_barrier = barrier_energies.max(dim=-1)[0]

        # 3. Barrier spread (variability)
        barrier_std = barrier_energies.std(dim=-1)

        derived = torch.stack([dg_rxn, max_barrier, barrier_std], dim=-1)

        return torch.cat([basic, derived], dim=-1)

    def forward(
        self,
        energy_predictions: Dict[str, Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """
        Classify mechanism based on predicted energy profiles.

        Args:
            energy_predictions: Dict mapping mechanism names to
                               (state_energies, barrier_energies) tuples

        Returns:
            Mechanism logits, shape (batch, n_mechanisms)
        """
        features = []
        for name in self.mechanism_names:
            if name in energy_predictions:
                state_e, barrier_e = energy_predictions[name]
                feat = self.extract_energy_features(state_e, barrier_e, self.mechanisms[name])
                features.append(feat)
            else:
                # Pad with zeros if mechanism not predicted
                batch_size = list(energy_predictions.values())[0][0].shape[0]
                device = list(energy_predictions.values())[0][0].device
                features.append(torch.zeros(batch_size, self.input_dims[name], device=device))

        all_features = torch.cat(features, dim=-1)
        h = self.mlp(all_features)
        logits = self.classifier(h)

        return logits


class HierarchicalClassifier(nn.Module):
    """
    Hierarchical classifier for mechanism discrimination.

    First classifies broad mechanism categories, then refines within categories.
    """

    def __init__(
        self,
        d_input: int,
        mechanism_hierarchy: Dict[str, List[str]],
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        """
        Args:
            d_input: Input dimension
            mechanism_hierarchy: Dict mapping category names to mechanism lists
                e.g., {"inhibition": ["competitive", "uncompetitive", "mixed"],
                       "multi_substrate": ["ordered_bi_bi", "random_bi_bi"]}
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        self.categories = list(mechanism_hierarchy.keys())
        self.mechanism_hierarchy = mechanism_hierarchy
        self.n_categories = len(self.categories)

        # Build flat mechanism list
        self.all_mechanisms = []
        self.mechanism_to_category = {}
        for cat, mechs in mechanism_hierarchy.items():
            for mech in mechs:
                self.all_mechanisms.append(mech)
                self.mechanism_to_category[mech] = cat

        # Category classifier
        self.category_classifier = MechanismClassifier(
            d_input=d_input,
            mechanism_names=self.categories,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Within-category classifiers
        self.sub_classifiers = nn.ModuleDict({
            cat: MechanismClassifier(
                d_input=d_input,
                mechanism_names=mechs,
                hidden_dims=[hidden_dims[-1]] if hidden_dims else [64],
                dropout=dropout,
            )
            for cat, mechs in mechanism_hierarchy.items()
            if len(mechs) > 1
        })

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        """
        Hierarchical classification.

        Args:
            h: Latent representation, shape (batch, d_input)

        Returns:
            Dict with keys:
                "category_logits": shape (batch, n_categories)
                "mechanism_logits": shape (batch, n_mechanisms)
                "<category>_logits": shape (batch, n_mechanisms_in_category)
        """
        result = {}

        # Category-level classification
        category_logits = self.category_classifier(h)
        result["category_logits"] = category_logits
        category_probs = F.softmax(category_logits, dim=-1)

        # Within-category classification
        mechanism_probs = []
        for cat in self.categories:
            mechs = self.mechanism_hierarchy[cat]
            if len(mechs) == 1:
                # Single mechanism in category - probability = category probability
                sub_probs = category_probs[:, self.categories.index(cat):self.categories.index(cat)+1]
            else:
                # Multiple mechanisms - classify within category
                sub_logits = self.sub_classifiers[cat](h)
                result[f"{cat}_logits"] = sub_logits
                sub_probs = F.softmax(sub_logits, dim=-1)
                # Weight by category probability
                sub_probs = sub_probs * category_probs[:, self.categories.index(cat):self.categories.index(cat)+1]

            mechanism_probs.append(sub_probs)

        # Combine all mechanism probabilities
        all_probs = torch.cat(mechanism_probs, dim=-1)

        # Convert back to logits for consistency
        result["mechanism_logits"] = torch.log(all_probs + 1e-10)

        return result

    def predict_mechanism(self, h: Tensor) -> Tuple[List[str], Tensor]:
        """
        Predict the most likely mechanism.

        Returns:
            Tuple of (mechanism_names, confidences)
        """
        outputs = self.forward(h)
        probs = F.softmax(outputs["mechanism_logits"], dim=-1)
        confidences, indices = probs.max(dim=-1)
        predicted_names = [self.all_mechanisms[i] for i in indices.tolist()]
        return predicted_names, confidences
