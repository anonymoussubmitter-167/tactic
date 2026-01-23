"""Tests for thermodynamic priors."""

import pytest
import torch
import numpy as np

from tactic_kinetics.training.thermodynamic_priors import (
    EnergyDistribution,
    ThermodynamicPriorExtractor,
)


class TestEnergyDistribution:
    """Tests for EnergyDistribution dataclass."""

    def test_distribution_creation(self):
        """Test creating an energy distribution."""
        dist = EnergyDistribution(
            mean=65.0,
            std=12.0,
            min_val=35.0,
            max_val=95.0,
            n_samples=1000,
            source="test",
        )

        assert dist.mean == 65.0
        assert dist.std == 12.0
        assert dist.min_val == 35.0
        assert dist.max_val == 95.0

    def test_distribution_sampling(self):
        """Test sampling from distribution."""
        dist = EnergyDistribution(
            mean=65.0,
            std=12.0,
            min_val=35.0,
            max_val=95.0,
            n_samples=1000,
            source="test",
        )

        samples = dist.sample(100)

        assert samples.shape == (100,)
        # Samples should be within bounds
        assert (samples >= dist.min_val).all()
        assert (samples <= dist.max_val).all()

    def test_distribution_sampling_reproducible(self):
        """Test that sampling is reproducible with same RNG."""
        dist = EnergyDistribution(
            mean=65.0,
            std=12.0,
            min_val=35.0,
            max_val=95.0,
            n_samples=1000,
            source="test",
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        samples1 = dist.sample(10, rng=rng1)
        samples2 = dist.sample(10, rng=rng2)

        np.testing.assert_array_equal(samples1, samples2)


class TestThermodynamicPriorExtractor:
    """Tests for ThermodynamicPriorExtractor."""

    def test_extractor_creates_distributions(self):
        """Test that extractor creates activation energy distributions."""
        # Create extractor without data files (uses defaults)
        extractor = ThermodynamicPriorExtractor(data_dir="nonexistent")

        # Should have activation energy distributions
        assert len(extractor.activation_energy_distributions) > 0
        assert "catalysis" in extractor.activation_energy_distributions
        assert "binding" in extractor.activation_energy_distributions
        assert "release" in extractor.activation_energy_distributions

    def test_catalysis_distribution_reasonable(self):
        """Test that catalysis distribution has reasonable values."""
        extractor = ThermodynamicPriorExtractor(data_dir="nonexistent")
        dist = extractor.activation_energy_distributions["catalysis"]

        # Catalysis barriers typically 40-100 kJ/mol
        assert 40 < dist.mean < 90
        assert dist.std > 5
        assert dist.min_val >= 20
        assert dist.max_val <= 120

    def test_binding_distribution_reasonable(self):
        """Test that binding distribution has reasonable values."""
        extractor = ThermodynamicPriorExtractor(data_dir="nonexistent")
        dist = extractor.activation_energy_distributions["binding"]

        # Binding barriers typically 20-60 kJ/mol
        assert 25 < dist.mean < 60
        assert dist.std > 3

    def test_release_distribution_reasonable(self):
        """Test that release distribution has reasonable values."""
        extractor = ThermodynamicPriorExtractor(data_dir="nonexistent")
        dist = extractor.activation_energy_distributions["release"]

        # Release barriers typically 30-70 kJ/mol
        assert 30 < dist.mean < 70
        assert dist.std > 3

    def test_get_activation_energy_prior(self):
        """Test getting activation energy prior by type."""
        extractor = ThermodynamicPriorExtractor(data_dir="nonexistent")

        catalysis_prior = extractor.get_activation_energy_prior("catalysis")
        binding_prior = extractor.get_activation_energy_prior("binding")

        assert catalysis_prior.mean != binding_prior.mean
        # Binding should have lower barrier than catalysis
        assert binding_prior.mean < catalysis_prior.mean

    def test_get_reaction_dg_prior_fallback(self):
        """Test reaction dG prior fallback when no data."""
        extractor = ThermodynamicPriorExtractor(data_dir="nonexistent")
        prior = extractor.get_reaction_dg_prior()

        # Should get fallback values
        assert prior is not None
        assert prior.source == "Fallback"

    def test_to_dict(self):
        """Test exporting distributions to dict."""
        extractor = ThermodynamicPriorExtractor(data_dir="nonexistent")
        data = extractor.to_dict()

        assert "activation_energies" in data
        assert "catalysis" in data["activation_energies"]
        assert "mean" in data["activation_energies"]["catalysis"]


class TestEyringEquation:
    """Tests for Eyring equation rate calculations."""

    def test_eyring_rate_from_barrier(self):
        """Test Eyring equation rate calculation."""
        from tactic_kinetics.models.ode_simulator import EnergyToRateConverter
        from tactic_kinetics.mechanisms.templates import get_mechanism_by_name

        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        converter = EnergyToRateConverter(mech)

        # Test with known barrier
        state_energies = torch.zeros(1, mech.n_energy_params)
        barrier = torch.ones(1, mech.n_barrier_params) * 60.0  # 60 kJ/mol

        forward_rates, _ = converter(state_energies, barrier)

        # At 60 kJ/mol barrier, rate should be around 10^2-10^4 s^-1
        assert forward_rates[0, 0] > 1  # Should be measurable
        assert forward_rates[0, 0] < 1e8  # Not too fast

    def test_temperature_dependence(self):
        """Test that higher temperature gives faster rates."""
        from tactic_kinetics.models.ode_simulator import EnergyToRateConverter
        from tactic_kinetics.mechanisms.templates import get_mechanism_by_name

        mech = get_mechanism_by_name("michaelis_menten_irreversible")

        converter_low = EnergyToRateConverter(mech, temperature=280.0)  # Cold
        converter_high = EnergyToRateConverter(mech, temperature=320.0)  # Warm

        state_energies = torch.zeros(1, mech.n_energy_params)
        barrier = torch.ones(1, mech.n_barrier_params) * 60.0

        rates_low, _ = converter_low(state_energies, barrier)
        rates_high, _ = converter_high(state_energies, barrier)

        # Higher temperature should give higher rates
        assert rates_high[0, 0] > rates_low[0, 0]

    def test_barrier_height_affects_rate(self):
        """Test that higher barriers give slower rates."""
        from tactic_kinetics.models.ode_simulator import EnergyToRateConverter
        from tactic_kinetics.mechanisms.templates import get_mechanism_by_name

        mech = get_mechanism_by_name("michaelis_menten_irreversible")
        converter = EnergyToRateConverter(mech)

        state_energies = torch.zeros(2, mech.n_energy_params)
        low_barrier = torch.ones(1, mech.n_barrier_params) * 50.0
        high_barrier = torch.ones(1, mech.n_barrier_params) * 70.0
        barriers = torch.cat([low_barrier, high_barrier], dim=0)

        rates, _ = converter(state_energies, barriers)

        # Lower barrier should give higher rate
        assert rates[0, 0] > rates[1, 0]
