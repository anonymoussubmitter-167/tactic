"""Tests for enzyme mechanism templates."""

import pytest
import torch
import numpy as np

from tactic_kinetics.mechanisms.base import (
    MechanismTemplate, State, Transition, StateType, TransitionType
)
from tactic_kinetics.mechanisms.templates import (
    get_all_mechanisms, get_mechanism_by_name, MECHANISM_REGISTRY
)


class TestMechanismTemplates:
    """Tests for mechanism templates."""

    def test_all_mechanisms_load(self):
        """Test that all 10 mechanisms load correctly."""
        mechanisms = get_all_mechanisms()
        assert len(mechanisms) == 10

    def test_mechanism_names(self):
        """Test that all expected mechanisms exist."""
        expected = [
            "michaelis_menten_irreversible",
            "michaelis_menten_reversible",
            "competitive_inhibition",
            "uncompetitive_inhibition",
            "mixed_inhibition",
            "substrate_inhibition",
            "ordered_bi_bi",
            "random_bi_bi",
            "ping_pong",
            "product_inhibition",
        ]
        mechanisms = get_all_mechanisms()
        for name in expected:
            assert name in mechanisms, f"Missing mechanism: {name}"

    def test_mechanism_has_reference_state(self):
        """Test that each mechanism has exactly one reference state."""
        for name, mech in get_all_mechanisms().items():
            ref_states = [s for s in mech.states if s.is_reference]
            assert len(ref_states) == 1, f"{name} has {len(ref_states)} reference states"

    def test_transition_types_annotated(self):
        """Test that all transitions have proper type annotations."""
        for name, mech in get_all_mechanisms().items():
            for trans in mech.transitions:
                assert trans.transition_type is not None
                assert isinstance(trans.transition_type, TransitionType)

    def test_michaelis_menten_structure(self):
        """Test Michaelis-Menten mechanism structure."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")

        assert mech.n_states == 3
        assert mech.n_transitions == 2
        assert len(mech.substrate_names) == 1
        assert len(mech.product_names) == 1
        assert "S" in mech.substrate_names
        assert "P" in mech.product_names

    def test_inhibition_mechanisms_have_inhibitor(self):
        """Test that inhibition mechanisms have inhibitor species."""
        inhibition_mechs = [
            "competitive_inhibition",
            "uncompetitive_inhibition",
            "mixed_inhibition",
        ]
        for name in inhibition_mechs:
            mech = get_mechanism_by_name(name)
            assert len(mech.inhibitor_names) > 0, f"{name} missing inhibitor"

    def test_bi_substrate_mechanisms_have_two_substrates(self):
        """Test that bi-substrate mechanisms have two substrates."""
        bi_substrate_mechs = ["ordered_bi_bi", "random_bi_bi", "ping_pong"]
        for name in bi_substrate_mechs:
            mech = get_mechanism_by_name(name)
            assert len(mech.substrate_names) == 2, f"{name} should have 2 substrates"
            assert len(mech.product_names) == 2, f"{name} should have 2 products"

    def test_energy_params_count(self):
        """Test energy parameter counting."""
        for name, mech in get_all_mechanisms().items():
            # n_energy_params should be n_states - 1 (excluding reference)
            assert mech.n_energy_params == mech.n_states - 1
            # n_barrier_params should equal n_transitions
            assert mech.n_barrier_params == mech.n_transitions

    def test_adjacency_matrix(self):
        """Test adjacency matrix generation."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        adj = mech.get_adjacency_matrix()

        assert adj.shape == (mech.n_states, mech.n_states)
        # Should have some non-zero entries (connections)
        assert adj.sum() > 0

    def test_mechanism_describe(self):
        """Test mechanism description generation."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        desc = mech.describe()

        assert "michaelis_menten_irreversible" in desc
        assert "States:" in desc
        assert "Transitions:" in desc


class TestTransitionTypes:
    """Tests for transition type annotations."""

    def test_binding_transitions(self):
        """Test that binding transitions are marked correctly."""
        for name, mech in get_all_mechanisms().items():
            for trans in mech.transitions:
                if "binding" in trans.name.lower():
                    assert trans.transition_type == TransitionType.BINDING

    def test_catalysis_transitions(self):
        """Test that catalysis transitions are marked correctly."""
        for name, mech in get_all_mechanisms().items():
            for trans in mech.transitions:
                if "catalysis" in trans.name.lower():
                    assert trans.transition_type == TransitionType.CATALYSIS

    def test_release_transitions(self):
        """Test that release transitions are marked correctly."""
        for name, mech in get_all_mechanisms().items():
            for trans in mech.transitions:
                if "release" in trans.name.lower():
                    assert trans.transition_type == TransitionType.RELEASE
