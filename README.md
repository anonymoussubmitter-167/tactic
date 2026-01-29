# TACTIC: Transformer Architecture for Classifying Thermodynamic and Inhibition Characteristics

TACTIC is a deep learning framework for classifying enzyme kinetic mechanisms from time-course experimental data. Given multi-condition kinetic trajectories, TACTIC identifies the underlying reaction mechanism from 10 canonical enzyme mechanisms spanning 5 mechanistic families.

## Key Results

| Method | Accuracy | Description |
|--------|----------|-------------|
| Random baseline | 10.0% | 10-class random guess |
| Classical (AIC) | 38.6% | Akaike Information Criterion model selection |
| **TACTIC v3** | **62.0%** | Multi-task transformer with cross-condition attention |

On real experimental data from 5 enzymes across 4 independent labs:
- **TACTIC v3**: 80% accuracy (4/5 correct), 85.4% average confidence
- **Classical AIC**: 0-20% accuracy, non-deterministic across runs

## Supported Mechanisms

TACTIC classifies 10 enzyme mechanisms organized into 5 families:

| Family | Mechanisms | Description |
|--------|------------|-------------|
| **Simple** | Michaelis-Menten Irreversible | E + S ⇌ ES → E + P |
| **Reversible** | MM Reversible, Product Inhibition | Reversible catalysis with product effects |
| **Inhibited** | Competitive, Uncompetitive, Mixed | Classic inhibition patterns |
| **Substrate-Regulated** | Substrate Inhibition | Excess substrate forms inactive ESS complex |
| **Bisubstrate** | Ordered Bi-Bi, Random Bi-Bi, Ping-Pong | Two-substrate mechanisms |

Family-level accuracy reaches **99.6%**, meaning errors occur within biochemically similar subtypes.

## Installation

```bash
git clone https://github.com/anonymoussubmitter-167/tactic.git
cd tactic
pip install -e .
```

### Dependencies

```bash
pip install torch numpy pandas scipy
pip install equilibrator-api  # Thermodynamic data
pip install zeep requests     # Database APIs
```

## Quick Start

```python
from tactic_kinetics import get_mechanism_by_name, get_all_mechanisms
from tactic_kinetics.models.multi_condition_classifier import create_multi_task_model

# List available mechanisms
mechanisms = get_all_mechanisms()
for name, mech in mechanisms.items():
    print(f"{name}: {mech.n_states} states, {mech.n_transitions} transitions")

# Create model
model = create_multi_task_model(
    n_mechanisms=10,
    d_model=128,
    n_heads=4,
    n_traj_layers=2,
    n_cross_layers=3,
)

# Load pretrained weights
checkpoint = torch.load('checkpoints/v3/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Training

### Generate Training Data

```bash
python train.py --version v3 --n-samples 80000 --generate-only
```

### Train Model

```bash
# v3 (recommended)
python train.py --version v3 --epochs 100 --batch-size 64 --lr 1e-4

# With multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --version v3 --epochs 100
```

### Model Versions

| Version | Architecture | Accuracy | Key Features |
|---------|--------------|----------|--------------|
| v0 | Single-curve baseline | 22% | One trajectory only |
| v1 | Basic multi-condition | 50% | 5 conditions, cross-attention |
| v2 | Improved multi-condition | 56% | 20 conditions, derived features, pairwise |
| v3 | Multi-task learning | 62% | Auxiliary heads for family, kinetics |

## Architecture

TACTIC uses a transformer-based architecture with:

1. **Trajectory Encoder**: Processes individual time-course curves with 1D convolutions and self-attention
2. **Condition Encoder**: Embeds experimental conditions (T, pH, [S]₀, [E]₀, [I]₀, etc.)
3. **Derived Feature Encoder**: Extracts kinetic signatures (v₀, t½, curvature, rate ratios)
4. **Cross-Condition Attention**: Compares how kinetics change across conditions
5. **Pairwise Comparison**: Explicit pairwise condition differences
6. **Classification Heads**: Mechanism prediction with optional auxiliary tasks

```
Input: N conditions × T timepoints × F features
  ↓
[Trajectory Encoder] → per-condition embeddings
  ↓
[Cross-Condition Attention] → mechanism-discriminative features
  ↓
[Classification Head] → mechanism probabilities
```

## Experiments

Run the full experiment suite:

```bash
python scripts/experiments/run_all_experiments.py
```

Individual experiments:

```bash
# Confidence calibration
python scripts/experiments/confidence_analysis.py

# Condition ablation (how many conditions needed?)
python scripts/experiments/condition_ablation.py

# Noise robustness
python scripts/experiments/noise_robustness.py

# Family-level accuracy
python scripts/experiments/family_accuracy.py

# Real enzyme data evaluation
python scripts/experiments/real_data_evaluation.py
```

## Key Findings

### Confidence Calibration
When TACTIC reports >90% confidence, it achieves **98% accuracy** (ECE = 0.064).

### Condition Requirements
- 1 condition: ~13% (insufficient)
- 5 conditions: ~33%
- 10 conditions: ~58%
- 20 conditions: ~62%

**Recommendation**: 7-10 experimental conditions provide the best accuracy/effort tradeoff.

### Noise Robustness
Only **4.1% accuracy degradation** at 30% measurement noise, suitable for real experimental data.

### Common Confusions
Errors occur between biochemically similar mechanisms:
- Competitive ↔ Uncompetitive inhibition
- Ordered Bi-Bi ↔ Ping-Pong
- MM Reversible ↔ Product Inhibition

These reflect fundamental ambiguities in the underlying kinetic patterns.

## Data Sources

TACTIC integrates with standard enzyme kinetics databases:

| Source | Data Type | Access |
|--------|-----------|--------|
| [eQuilibrator](https://equilibrator.weizmann.ac.il/) | Thermodynamic ΔG values | Python API |
| [BRENDA](https://www.brenda-enzymes.org/) | Km, kcat, Ki parameters | SOAP API |
| [SABIO-RK](http://sabiork.h-its.org/) | Kinetic rate laws | REST API |
| [EnzymeML](https://enzymeml.org/) | Standardized kinetic data | OMEX files |

See [DATA_SOURCES.md](DATA_SOURCES.md) for detailed access instructions.

## Project Structure

```
tactic/
├── tactic_kinetics/
│   ├── mechanisms/         # Mechanism templates and definitions
│   │   ├── base.py         # MechanismTemplate, State, Transition classes
│   │   └── templates.py    # 10 predefined mechanisms
│   ├── models/
│   │   ├── multi_condition_classifier.py  # v0-v3 model architectures
│   │   ├── encoder.py      # Trajectory and condition encoders
│   │   └── ode_simulator.py # ODE-based kinetic simulation
│   ├── training/
│   │   ├── multi_condition_generator.py  # Synthetic data generation
│   │   ├── multi_condition_dataset.py    # PyTorch dataset
│   │   └── thermodynamic_priors.py       # Thermodynamic constraints
│   ├── data/               # Data loaders for external sources
│   ├── benchmarks/         # Benchmark utilities
│   └── utils/              # Thermodynamic calculations, constants
├── scripts/
│   ├── experiments/        # Experiment scripts
│   └── data/               # Data acquisition scripts
├── data/
│   └── real/               # Real enzyme datasets
├── checkpoints/            # Trained model weights
├── results/                # Experiment results and analysis
├── examples/               # Usage examples
└── tests/                  # Unit tests
```

## Citation

If you use TACTIC in your research, please cite:

```bibtex
@article{tactic2024,
  title={TACTIC: Transformer Architecture for Classifying Thermodynamic and Inhibition Characteristics of Enzyme Mechanisms},
  author={Anonymous},
  journal={Submitted},
  year={2024}
}
```

## License

MIT License
