# TACTIC: Thermodynamically-Native Active Transfer Inference for Multi-Condition Enzyme Kinetics

TACTIC is a deep learning framework for classifying enzyme kinetic mechanisms directly from multi-condition kinetic data. Our key insight is that mechanism discrimination requires comparing how kinetics change across experimental conditions—single kinetic curves are fundamentally insufficient.

## Key Results

| Method | Accuracy | Description |
|--------|----------|-------------|
| Random baseline | 10.0% | 10-class random guess |
| Classical (AIC) | 38.6% | Akaike Information Criterion model selection |
| **TACTIC v3** | **62.0%** | +23.4 percentage points vs classical |

**Real experimental data** (5 enzymes, 4 independent labs):
- **TACTIC**: 80% accuracy (4/5 correct), deterministic predictions
- **Classical AIC**: 0-20% accuracy, non-deterministic across runs

TACTIC is **134× faster** than classical model fitting.

## The Multi-Condition Requirement

Single kinetic curves cannot discriminate mechanisms. Consider competitive vs uncompetitive inhibition—both reduce velocity at any single inhibitor concentration. The discriminative signal emerges only by comparing how kinetics change across conditions:

- **Competitive**: High substrate overcomes inhibition (Km increases, Vmax unchanged)
- **Uncompetitive**: High substrate worsens inhibition (both Km and Vmax decrease)

Our experiments confirm: 1 condition achieves ~12.8% accuracy (near random); 7-10 conditions are needed for reliable classification.

## Supported Mechanisms

TACTIC classifies 10 enzyme mechanisms organized into 5 families:

| Family | Mechanisms | Per-Mechanism Accuracy |
|--------|------------|------------------------|
| **Simple** | MM Irreversible | 99% |
| **Reversible** | MM Reversible, Product Inhibition | 54%, 82% |
| **Inhibited** | Competitive, Uncompetitive, Mixed | 61%, 61%, 36% |
| **Substrate-Regulated** | Substrate Inhibition | 97% |
| **Bisubstrate** | Ordered Bi-Bi, Random Bi-Bi, Ping-Pong | 32%, 41%, 57% |

**Family-level accuracy: 99.6%** — errors occur within biochemically similar subtypes, not between mechanistically different types.

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
    d_model=256,
    n_heads=8,
    n_traj_layers=3,
    n_cross_layers=3,
)

# Load pretrained weights
checkpoint = torch.load('checkpoints/v3/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Training

### Generate Training Data

```bash
python train.py --version v3 --n-samples 50000 --generate-only
```

### Train Model

```bash
# v3 (recommended)
python train.py --version v3 --epochs 50 --batch-size 32 --lr 1e-3

# With multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --version v3 --epochs 50
```

### Model Versions

| Version | Key Features | Accuracy | vs Classical |
|---------|--------------|----------|--------------|
| v0 | Single curve only | 10.6% | -28.0% |
| v1 | Multi-condition (5 cond) | 50.1% | +11.5% |
| v2 | +Derived features (20 cond) | 56.1% | +17.5% |
| v3 | +Auxiliary losses | **62.0%** | **+23.4%** |

## Architecture

TACTIC uses a transformer-based architecture (~2.1M parameters):

1. **Trajectory Encoder**: Conv1D (kernels 3,5,7) + 3-layer Transformer with positional embeddings
2. **Condition Encoder**: MLP to embed experimental parameters ([S]₀, [E]₀, [I], temperature)
3. **Derived Feature Encoder**: Kinetic signatures (v₀, t½, final conversion)
4. **Cross-Condition Attention**: 3-layer Transformer enabling trajectories to attend to each other
5. **Pairwise Comparison**: MLP + 2-layer Self-attention
6. **Classification Head**: MLP (256→128→10)

```
Input: K conditions × T timepoints (K ∈ [5,20], T ∈ [20,50])
  ↓
[Trajectory Encoder] → per-condition embeddings (d=256)
  ↓
[Cross-Condition Attention] → learns which condition comparisons are informative
  ↓
[Attention-Weighted Pooling + MLP] → mechanism probabilities
```

**Ablation results**: Cross-attention is critical (−7.8% without it), confirming that mechanism discrimination requires explicit comparison across conditions.

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
TACTIC is well-calibrated (ECE = 0.064). When confidence >90%, accuracy is **98%**, enabling reliable triage of uncertain cases.

### Condition Requirements
| Conditions | Accuracy |
|------------|----------|
| 1 | 12.8% (≈ random) |
| 5 | ~33% |
| 7 | ~50% |
| 10 | ~58% |
| 20 | 62% |

**Recommendation**: 7-10 experimental conditions provide the best accuracy/effort tradeoff.

### Noise Robustness
Only **4.1% accuracy degradation** at 30% measurement noise—suitable for real experimental data with typical 5-15% error.

### Confusion Patterns
Errors align with theoretical identifiability constraints:

| Confused Pair | Mutual Confusion Rate |
|---------------|----------------------|
| Competitive ↔ Uncompetitive | 32.5% |
| Ordered Bi-Bi ↔ Ping-Pong | 32.5% |
| MM Reversible ↔ Product Inhibition | 32.0% |
| Ordered Bi-Bi ↔ Random Bi-Bi | 31.0% |
| Competitive ↔ Mixed | 24.5% |

These reflect fundamental biochemical ambiguities—mechanisms non-identifiable under single conditions remain challenging even with multi-condition data.

## Data Sources

Training data is generated by numerically integrating mechanistic ODEs with parameters sampled from experimental databases:

| Source | Data Type | Access |
|--------|-----------|--------|
| [BRENDA](https://www.brenda-enzymes.org/) | Km, kcat, Ki parameters | SOAP API |
| [SABIO-RK](http://sabiork.h-its.org/) | Kinetic rate laws | REST API |
| [eQuilibrator](https://equilibrator.weizmann.ac.il/) | Thermodynamic ΔG values | Python API |
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

## License

MIT License
