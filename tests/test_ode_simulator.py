"""Tests for ODE simulator."""

import pytest
import torch
import numpy as np

from tactic_kinetics.mechanisms.templates import get_mechanism_by_name, get_all_mechanisms
from tactic_kinetics.models.ode_simulator import (
    GeneralMechanismODE,
    EnergyToRateConverter,
    simulate_general_mechanism,
    get_initial_conditions,
    simulate_michaelis_menten,
)


class TestEnergyToRateConverter:
    """Tests for energy to rate conversion."""

    def test_converter_output_shapes(self):
        """Test that converter outputs have correct shapes."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        converter = EnergyToRateConverter(mech)

        batch_size = 4
        state_energies = torch.randn(batch_size, mech.n_energy_params)
        barrier_energies = torch.randn(batch_size, mech.n_barrier_params) + 60.0

        forward_rates, reverse_rates = converter(state_energies, barrier_energies)

        assert forward_rates.shape == (batch_size, mech.n_transitions)
        assert reverse_rates.shape == (batch_size, mech.n_transitions)

    def test_converter_positive_rates(self):
        """Test that rates are positive."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        converter = EnergyToRateConverter(mech)

        state_energies = torch.randn(2, mech.n_energy_params)
        barrier_energies = torch.randn(2, mech.n_barrier_params) + 60.0

        forward_rates, reverse_rates = converter(state_energies, barrier_energies)

        assert (forward_rates > 0).all()
        assert (reverse_rates >= 0).all()

    def test_converter_lower_barrier_higher_rate(self):
        """Test that lower barriers give higher rates."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        converter = EnergyToRateConverter(mech)

        state_energies = torch.zeros(2, mech.n_energy_params)
        barrier_low = torch.ones(1, mech.n_barrier_params) * 50.0
        barrier_high = torch.ones(1, mech.n_barrier_params) * 70.0

        barrier_energies = torch.cat([barrier_low, barrier_high], dim=0)
        forward_rates, _ = converter(state_energies, barrier_energies)

        # Lower barrier should give higher rate
        assert forward_rates[0, 0] > forward_rates[1, 0]


class TestGeneralMechanismODE:
    """Tests for general mechanism ODE."""

    def test_ode_species_tracking(self):
        """Test that ODE tracks correct species."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        ode = GeneralMechanismODE(mech)

        # Should track E, ES, S, P
        assert "E" in ode.species_idx
        assert "ES" in ode.species_idx
        assert "S" in ode.species_idx
        assert "P" in ode.species_idx

    def test_ode_inhibition_species(self):
        """Test that inhibition mechanisms track inhibitor."""
        mech = get_mechanism_by_name("competitive_inhibition")
        ode = GeneralMechanismODE(mech)

        assert "I" in ode.species_idx
        assert "EI" in ode.species_idx

    def test_ode_bi_substrate_species(self):
        """Test that bi-substrate mechanisms track both substrates."""
        mech = get_mechanism_by_name("ordered_bi_bi")
        ode = GeneralMechanismODE(mech)

        assert "A" in ode.species_idx
        assert "B" in ode.species_idx
        assert "P" in ode.species_idx
        assert "Q" in ode.species_idx


class TestODESimulation:
    """Tests for ODE simulation."""

    @pytest.mark.parametrize("mech_name", list(get_all_mechanisms().keys()))
    def test_simulation_runs_for_all_mechanisms(self, mech_name):
        """Test that simulation runs for all mechanisms without errors."""
        mech = get_mechanism_by_name(mech_name)

        batch_size = 2
        state_energies = torch.randn(batch_size, mech.n_energy_params) * 5.0
        barrier_energies = torch.ones(batch_size, mech.n_barrier_params) * 60.0

        converter = EnergyToRateConverter(mech)
        forward_rates, reverse_rates = converter(state_energies, barrier_energies)

        E0 = torch.ones(batch_size) * 0.01
        substrate_concs = {sub: torch.ones(batch_size) * 1.0 for sub in mech.substrate_names}
        init_conds = get_initial_conditions(mech, E0, substrate_concs, None)

        t_eval = torch.linspace(0, 10, 10)

        trajectories = simulate_general_mechanism(
            mech, forward_rates, reverse_rates, init_conds, t_eval
        )

        # Check no NaN values
        for species, traj in trajectories.items():
            assert not torch.isnan(traj).any(), f"NaN in {mech_name} trajectory for {species}"

    def test_product_increases_over_time(self):
        """Test that product concentration increases (for irreversible MM)."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")

        batch_size = 2
        state_energies = torch.zeros(batch_size, mech.n_energy_params)
        barrier_energies = torch.ones(batch_size, mech.n_barrier_params) * 60.0

        converter = EnergyToRateConverter(mech)
        forward_rates, reverse_rates = converter(state_energies, barrier_energies)

        E0 = torch.ones(batch_size) * 0.01
        substrate_concs = {"S": torch.ones(batch_size) * 1.0}
        init_conds = get_initial_conditions(mech, E0, substrate_concs, None)

        t_eval = torch.linspace(0, 100, 20)

        trajectories = simulate_general_mechanism(
            mech, forward_rates, reverse_rates, init_conds, t_eval
        )

        # Product should increase
        P = trajectories["P"]
        assert P[:, -1].mean() > P[:, 0].mean()

    def test_substrate_decreases_over_time(self):
        """Test that substrate concentration decreases."""
        mech = get_mechanism_by_name("michaelis_menten_irreversible")

        batch_size = 2
        state_energies = torch.zeros(batch_size, mech.n_energy_params)
        barrier_energies = torch.ones(batch_size, mech.n_barrier_params) * 60.0

        converter = EnergyToRateConverter(mech)
        forward_rates, reverse_rates = converter(state_energies, barrier_energies)

        E0 = torch.ones(batch_size) * 0.01
        substrate_concs = {"S": torch.ones(batch_size) * 1.0}
        init_conds = get_initial_conditions(mech, E0, substrate_concs, None)

        t_eval = torch.linspace(0, 100, 20)

        trajectories = simulate_general_mechanism(
            mech, forward_rates, reverse_rates, init_conds, t_eval
        )

        # Substrate should decrease
        S = trajectories["S"]
        assert S[:, -1].mean() < S[:, 0].mean()


class TestSimpleMichaelisMenten:
    """Tests for simple MM simulation."""

    def test_simple_mm_runs(self):
        """Test that simple MM simulation runs."""
        kcat = torch.ones(4) * 10.0
        Km = torch.ones(4) * 0.5
        E_total = torch.ones(4) * 0.01
        S0 = torch.ones(4) * 1.0
        t_eval = torch.linspace(0, 100, 50)

        S, P = simulate_michaelis_menten(kcat, Km, E_total, S0, t_eval)

        assert S.shape == (4, 50)
        assert P.shape == (4, 50)
        assert not torch.isnan(S).any()
        assert not torch.isnan(P).any()

    def test_mass_conservation(self):
        """Test that S + P is approximately conserved."""
        kcat = torch.ones(2) * 10.0
        Km = torch.ones(2) * 0.5
        E_total = torch.ones(2) * 0.01
        S0 = torch.ones(2) * 1.0
        t_eval = torch.linspace(0, 100, 50)

        S, P = simulate_michaelis_menten(kcat, Km, E_total, S0, t_eval)

        # S + P should approximately equal S0
        total = S + P
        assert torch.allclose(total, S0.unsqueeze(-1).expand_as(total), atol=0.01)
