"""Tests for data loaders."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import pandas as pd

from tactic_kinetics.data.brenda_loader import (
    km_to_binding_dg,
    kcat_to_activation_dg,
    ki_to_binding_dg,
    KineticParameter,
    EnzymeKinetics,
)
from tactic_kinetics.data.fairdom_loader import (
    FAIRDOMLoader,
    KineticsExperiment,
    KineticsDataset,
    FAIRDOMTorchDataset,
)


class TestThermodynamicConversions:
    """Tests for kinetic parameter to energy conversions."""

    def test_km_to_binding_dg_reasonable_range(self):
        """Test Km conversion gives reasonable binding energies."""
        # Typical Km values: 0.01 - 10 mM
        km_values = [0.01, 0.1, 1.0, 10.0]

        for km in km_values:
            dg = km_to_binding_dg(km)
            # Binding energies should typically be -10 to +10 kJ/mol
            assert -50 < dg < 50, f"Km={km}mM gave dG={dg} kJ/mol"

    def test_km_lower_is_tighter_binding(self):
        """Test that lower Km gives more negative (tighter) binding energy."""
        km_low = 0.01  # mM - tight binding
        km_high = 10.0  # mM - weak binding

        dg_low = km_to_binding_dg(km_low)
        dg_high = km_to_binding_dg(km_high)

        # Lower Km should give more negative dG
        assert dg_low < dg_high

    def test_kcat_to_activation_dg_reasonable_range(self):
        """Test kcat conversion gives reasonable activation energies."""
        # Typical kcat values: 0.1 - 10000 s^-1
        kcat_values = [0.1, 1.0, 10.0, 100.0, 1000.0]

        for kcat in kcat_values:
            dg = kcat_to_activation_dg(kcat)
            # Activation barriers typically 40-100 kJ/mol
            assert 30 < dg < 120, f"kcat={kcat}s^-1 gave dG‡={dg} kJ/mol"

    def test_kcat_higher_is_lower_barrier(self):
        """Test that higher kcat gives lower activation barrier."""
        kcat_slow = 0.1  # s^-1
        kcat_fast = 1000.0  # s^-1

        dg_slow = kcat_to_activation_dg(kcat_slow)
        dg_fast = kcat_to_activation_dg(kcat_fast)

        # Faster enzyme should have lower barrier
        assert dg_fast < dg_slow

    def test_ki_to_binding_dg_reasonable_range(self):
        """Test Ki conversion gives reasonable inhibitor binding energies."""
        ki_values = [0.001, 0.01, 0.1, 1.0]

        for ki in ki_values:
            dg = ki_to_binding_dg(ki)
            # Inhibitor binding typically -40 to +10 kJ/mol
            assert -60 < dg < 30, f"Ki={ki}mM gave dG={dg} kJ/mol"

    def test_ki_lower_is_tighter_inhibition(self):
        """Test that lower Ki gives more negative (tighter) inhibitor binding."""
        ki_strong = 0.001  # mM - strong inhibitor
        ki_weak = 1.0  # mM - weak inhibitor

        dg_strong = ki_to_binding_dg(ki_strong)
        dg_weak = ki_to_binding_dg(ki_weak)

        # Strong inhibitor has more negative binding energy
        assert dg_strong < dg_weak


class TestKineticParameterDataclasses:
    """Tests for kinetic parameter data structures."""

    def test_kinetic_parameter_creation(self):
        """Test creating KineticParameter."""
        param = KineticParameter(
            ec_number="1.1.1.1",
            organism="E. coli",
            value=0.5,
            substrate="Test substrate",
        )

        assert param.ec_number == "1.1.1.1"
        assert param.value == 0.5
        assert param.substrate == "Test substrate"

    def test_kinetic_parameter_with_range(self):
        """Test KineticParameter with value range."""
        param = KineticParameter(
            ec_number="1.1.1.1",
            organism="E. coli",
            value=0.5,
            value_max=1.0,
        )

        assert param.has_uncertainty == True

    def test_enzyme_kinetics_creation(self):
        """Test creating EnzymeKinetics."""
        km_param = KineticParameter(
            ec_number="1.1.1.1",
            organism="E. coli",
            value=0.5,
        )
        kcat_param = KineticParameter(
            ec_number="1.1.1.1",
            organism="E. coli",
            value=100.0,
        )

        kinetics = EnzymeKinetics(
            ec_number="1.1.1.1",
            organism="E. coli",
            km_values=[km_param],
            kcat_values=[kcat_param],
        )

        assert kinetics.ec_number == "1.1.1.1"
        assert len(kinetics.km_values) == 1
        assert len(kinetics.kcat_values) == 1

    def test_enzyme_kinetics_best_values(self):
        """Test getting best Km/kcat values."""
        km_params = [
            KineticParameter(ec_number="1.1.1.1", organism="E. coli", value=0.3),
            KineticParameter(ec_number="1.1.1.1", organism="E. coli", value=0.5),
            KineticParameter(ec_number="1.1.1.1", organism="E. coli", value=0.7),
        ]

        kinetics = EnzymeKinetics(
            ec_number="1.1.1.1",
            km_values=km_params,
        )

        best_km = kinetics.get_best_km()
        assert best_km == 0.5  # Median


class TestKineticsExperiment:
    """Tests for KineticsExperiment dataclass."""

    def test_experiment_creation(self):
        """Test creating KineticsExperiment."""
        times = np.linspace(0, 100, 20)
        concs = 1 - np.exp(-0.05 * times)  # Product formation curve

        exp = KineticsExperiment(
            experiment_id="test_001",
            enzyme_name="Test Enzyme",
            substrate_name="Test Substrate",
            times=times,
            concentrations=concs,
            temperature=298.15,
            ph=7.0,
        )

        assert exp.experiment_id == "test_001"
        assert exp.n_timepoints == 20
        assert exp.temperature == 298.15

    def test_experiment_to_tensors(self):
        """Test conversion to tensors."""
        times = np.array([0, 1, 2, 3, 4])
        concs = np.array([0, 0.2, 0.4, 0.6, 0.8])

        exp = KineticsExperiment(
            experiment_id="test",
            enzyme_name="E",
            substrate_name="S",
            times=times,
            concentrations=concs,
        )

        t_tensor, c_tensor = exp.to_tensors()

        assert isinstance(t_tensor, torch.Tensor)
        assert isinstance(c_tensor, torch.Tensor)
        assert t_tensor.shape == (5,)
        assert c_tensor.shape == (5,)


class TestKineticsDataset:
    """Tests for KineticsDataset."""

    def test_dataset_creation(self):
        """Test creating KineticsDataset."""
        exp1 = KineticsExperiment(
            experiment_id="exp1",
            enzyme_name="Enzyme A",
            substrate_name="Substrate X",
            times=np.array([0, 1, 2]),
            concentrations=np.array([0, 0.5, 0.8]),
        )
        exp2 = KineticsExperiment(
            experiment_id="exp2",
            enzyme_name="Enzyme B",
            substrate_name="Substrate Y",
            times=np.array([0, 1, 2]),
            concentrations=np.array([0, 0.3, 0.6]),
        )

        dataset = KineticsDataset(experiments=[exp1, exp2])

        assert len(dataset) == 2
        assert dataset[0].enzyme_name == "Enzyme A"
        assert dataset[1].enzyme_name == "Enzyme B"

    def test_filter_by_enzyme(self):
        """Test filtering by enzyme name."""
        exp1 = KineticsExperiment(
            experiment_id="exp1",
            enzyme_name="Lactase",
            substrate_name="Lactose",
            times=np.array([0, 1]),
            concentrations=np.array([0, 0.5]),
        )
        exp2 = KineticsExperiment(
            experiment_id="exp2",
            enzyme_name="Amylase",
            substrate_name="Starch",
            times=np.array([0, 1]),
            concentrations=np.array([0, 0.3]),
        )

        dataset = KineticsDataset(experiments=[exp1, exp2])
        filtered = dataset.filter_by_enzyme("lactase")

        assert len(filtered) == 1
        assert filtered[0].enzyme_name == "Lactase"

    def test_filter_by_substrate(self):
        """Test filtering by substrate name."""
        exp1 = KineticsExperiment(
            experiment_id="exp1",
            enzyme_name="Enzyme A",
            substrate_name="ATP",
            times=np.array([0, 1]),
            concentrations=np.array([0, 0.5]),
        )
        exp2 = KineticsExperiment(
            experiment_id="exp2",
            enzyme_name="Enzyme B",
            substrate_name="GTP",
            times=np.array([0, 1]),
            concentrations=np.array([0, 0.3]),
        )

        dataset = KineticsDataset(experiments=[exp1, exp2])
        filtered = dataset.filter_by_substrate("atp")

        assert len(filtered) == 1
        assert filtered[0].substrate_name == "ATP"


class TestFAIRDOMTorchDataset:
    """Tests for FAIRDOMTorchDataset."""

    def test_torch_dataset_creation(self):
        """Test creating PyTorch dataset."""
        exp = KineticsExperiment(
            experiment_id="test",
            enzyme_name="Enzyme",
            substrate_name="Substrate",
            times=np.linspace(0, 100, 30),
            concentrations=np.linspace(0, 1, 30),
            temperature=298.15,
            ph=7.0,
            substrate_conc=1.0,
            enzyme_conc=0.01,
        )
        kinetics_dataset = KineticsDataset(experiments=[exp])

        torch_dataset = FAIRDOMTorchDataset(kinetics_dataset, max_timepoints=50)

        assert len(torch_dataset) == 1

    def test_torch_dataset_getitem(self):
        """Test getting item from PyTorch dataset."""
        times = np.linspace(0, 100, 30)
        exp = KineticsExperiment(
            experiment_id="test",
            enzyme_name="Enzyme",
            substrate_name="Substrate",
            times=times,
            concentrations=np.linspace(0, 1, 30),
            temperature=298.15,
            ph=7.0,
            substrate_conc=1.0,
            enzyme_conc=0.01,
        )
        kinetics_dataset = KineticsDataset(experiments=[exp])
        torch_dataset = FAIRDOMTorchDataset(kinetics_dataset, max_timepoints=50)

        item = torch_dataset[0]

        assert "times" in item
        assert "values" in item
        assert "mask" in item
        assert "conditions" in item
        assert item["times"].shape == (50,)
        assert item["values"].shape == (50,)
        assert item["mask"].shape == (50,)
        assert item["conditions"].shape == (4,)

    def test_torch_dataset_padding(self):
        """Test that short sequences are padded correctly."""
        exp = KineticsExperiment(
            experiment_id="test",
            enzyme_name="Enzyme",
            substrate_name="Substrate",
            times=np.array([0, 1, 2, 3, 4]),  # Only 5 points
            concentrations=np.array([0, 0.2, 0.4, 0.6, 0.8]),
        )
        kinetics_dataset = KineticsDataset(experiments=[exp])
        torch_dataset = FAIRDOMTorchDataset(kinetics_dataset, max_timepoints=10)

        item = torch_dataset[0]

        # First 5 should be valid
        assert item["mask"][:5].all()
        # Last 5 should be padding
        assert not item["mask"][5:].any()


class TestFAIRDOMLoader:
    """Tests for FAIRDOMLoader."""

    def test_loader_creation(self):
        """Test creating FAIRDOMLoader."""
        loader = FAIRDOMLoader()
        assert loader.data_dir == Path("data/fairdom")

        loader = FAIRDOMLoader(data_dir="/custom/path")
        assert loader.data_dir == Path("/custom/path")

    def test_load_csv(self):
        """Test loading CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time,Product\n")
            f.write("0,0.0\n")
            f.write("10,0.2\n")
            f.write("20,0.4\n")
            f.write("30,0.6\n")
            temp_path = f.name

        try:
            loader = FAIRDOMLoader()
            dataset = loader.load_csv(temp_path)

            assert len(dataset) == 1
            assert dataset[0].n_timepoints == 4
        finally:
            Path(temp_path).unlink()

    def test_load_excel(self):
        """Test loading Excel file."""
        pytest.importorskip("openpyxl")

        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name

        try:
            df = pd.DataFrame({
                "Time": [0, 10, 20, 30, 40],
                "Product": [0.0, 0.2, 0.4, 0.6, 0.7],
            })
            df.to_excel(temp_path, index=False)

            loader = FAIRDOMLoader()
            dataset = loader.load_excel(temp_path)

            assert len(dataset) == 1
            assert dataset[0].n_timepoints == 5
        finally:
            Path(temp_path).unlink()
