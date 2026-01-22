# TACTIC-Kinetics Training Results

All training results for the enzyme mechanism classification model.

---

## Results Summary

| Version | Approach | Accuracy | Date |
|---------|----------|----------|------|
| v0 | Single-curve training | 22% | - |
| v1 | Multi-condition (5 conditions) | 57% | - |
| v2 | Multi-condition + improved strategies | 65% | 2026-01-19 |
| v3 | Multi-task learning (auxiliary heads) | 66% | 2026-01-21 |

---

## v0: Single-Curve Training (Baseline)

**Accuracy: 22%**

### Problem
Single kinetic curves cannot distinguish between mechanisms that have similar steady-state behavior but differ in how parameters change with conditions.

### Key Insight
Some mechanisms are mathematically indistinguishable from single S(t) curves:
- Competitive vs uncompetitive vs mixed inhibition require [I] variation
- Ordered vs random bi-bi require [A] and [B] variation

The 22% accuracy reflects the model learning ~3-4 distinguishable mechanism families rather than all 10 individual mechanisms.

---

## v1: Multi-Condition Training

**Accuracy: 57%** (+35% from v0)

### Approach
One sample = one enzyme measured under multiple experimental conditions (5 conditions per sample).

### Architecture
- TrajectoryEncoder: 1D convolutions + transformer for each trajectory
- ConditionEncoder: Encodes experimental conditions
- Cross-attention: Compares trajectories across conditions
- Attention pooling + classification head

### Per-Mechanism Results

| Mechanism | Accuracy | Notes |
|-----------|----------|-------|
| michaelis_menten_irreversible | 99.5% | ✓ |
| michaelis_menten_reversible | 43.0% | Confused with product_inhib |
| competitive_inhibition | 45.2% | Confused with uncompetitive |
| uncompetitive_inhibition | 62.4% | |
| mixed_inhibition | 24.7% | Confused with comp/uncomp |
| substrate_inhibition | 100.0% | ✓ |
| ordered_bi_bi | 3.0% | Confused with random/ping_pong |
| random_bi_bi | 41.6% | |
| ping_pong | 60.4% | |
| product_inhibition | 74.0% | |

### Confusion Analysis
- Bi-substrate mechanisms nearly indistinguishable (ordered_bi_bi at 3%)
- Inhibition types mixing up (competitive ↔ uncompetitive ↔ mixed)
- Condition variation strategies too narrow

---

## v2: Improved Condition Strategies

**Accuracy: 65%** (+8% from v1)

### Changes from v1

1. **Better bi-substrate variation**: 4 slices through [A]-[B] space (20 conditions)
   - Fix [B] at low/high, vary [A] widely
   - Fix [A] at low/high, vary [B] widely
   - Reveals intersection pattern for discrimination

2. **Full [I] × [S] grid for inhibition**: 4 [I] levels × 5 [S] levels = 20 conditions
   - Enables measurement of Km_app and Vmax_app changes with [I]

3. **Track multiple species**: Now track both S and P (5 trajectory features)

4. **Derived kinetic features**: v0, t_half, rate_ratio, final_conversion, etc.

5. **Pairwise comparison module**: Explicit comparison between condition pairs

### Per-Mechanism Results

| Mechanism | v1 | v2 | Change |
|-----------|-----|-----|--------|
| michaelis_menten_irreversible | 99.5% | 100.0% | +0.5% |
| michaelis_menten_reversible | 43.0% | 61.3% | **+18.3%** |
| competitive_inhibition | 45.2% | 68.6% | **+23.4%** |
| uncompetitive_inhibition | 62.4% | 54.1% | -8.3% |
| mixed_inhibition | 24.7% | 38.1% | +13.4% |
| substrate_inhibition | 100.0% | 99.5% | -0.5% |
| ordered_bi_bi | 3.0% | 37.7% | **+34.7%** |
| random_bi_bi | 41.6% | 51.1% | +9.5% |
| ping_pong | 60.4% | 46.4% | -14.0% |
| product_inhibition | 74.0% | 82.7% | +8.7% |

### Confusion Matrix

```
                          mich mich comp unco mixe subs orde rand ping prod
michaelis_menten_irrevers[205]   0    0    0    0    0    0    0    0    0
michaelis_menten_reversib   0 [114]   0    0    0    0    0    0    0   72
competitive_inhibition      0    0 [144]  50   16    0    0    0    0    0
uncompetitive_inhibition    0    0   85 [111]   9    0    0    0    0    0
mixed_inhibition            0    0   74   46 [ 74]   0    0    0    0    0
substrate_inhibition        1    0    0    0    0 [207]   0    0    0    0
ordered_bi_bi               0    0    0    0    0    0 [ 75]  75   49    0
random_bi_bi                0    0    0    0    0    0   47 [ 97]  46    0
ping_pong                   0    0    0    0    0    0   41   70 [ 96]   0
product_inhibition          0   34    0    0    0    0    0    0    0 [162]
```

### Most Confused Pairs

| True | Predicted | Count |
|------|-----------|-------|
| uncompetitive_inhibition | competitive_inhibition | 85 |
| ordered_bi_bi | random_bi_bi | 75 |
| mixed_inhibition | competitive_inhibition | 74 |
| michaelis_menten_reversible | product_inhibition | 72 |
| ping_pong | random_bi_bi | 70 |

### Training Dynamics

Model converged around epoch 60-65, plateaued at 63-65% for final 50 epochs.

```
Epochs 51-75:  63.5% avg
Epochs 76-100: 64.3% avg
Change: +0.7% (plateaued)
```

---

## v3: Multi-Task Learning

**Accuracy: 65.9%** (+0.9% from v2)

### Hypothesis
Adding auxiliary prediction tasks (kinetic parameters, parameter change patterns) would force the model to learn mechanism-discriminative features.

### Changes from v2

1. **MultiTaskClassifier with auxiliary heads**:
   - `KineticParamHead`: Predicts Vmax_app, Km_app per condition
   - `PatternHead`: Predicts dVmax/dI, dKm/dI slopes (discriminates inhibition types)

2. **Auxiliary targets in dataset**:
   - Per-condition: v0_normalized, Km_estimate (log scale)
   - Cross-condition: Parameter change slopes with [I] and [S]

3. **Curriculum learning**: Auxiliary weight decays 0.5 → 0.1 over training

4. **Longer simulation for reversible mechanisms**: 5x time extension for MM_reversible and product_inhibition

### Per-Mechanism Results

| Mechanism | v2 | v3 | Change |
|-----------|-----|-----|--------|
| michaelis_menten_irreversible | 100.0% | 99.5% | -0.5% |
| michaelis_menten_reversible | 61.3% | 59.1% | -2.2% |
| competitive_inhibition | 68.6% | 63.3% | -5.3% |
| uncompetitive_inhibition | 54.1% | 61.5% | **+7.4%** |
| mixed_inhibition | 38.1% | 41.8% | +3.7% |
| substrate_inhibition | 99.5% | 98.6% | -0.9% |
| ordered_bi_bi | 37.7% | 39.7% | +2.0% |
| random_bi_bi | 51.1% | 50.5% | -0.6% |
| ping_pong | 46.4% | 56.5% | **+10.1%** |
| product_inhibition | 82.7% | 82.7% | 0% |

### Confusion Matrix

```
                          mich mich comp unco mixe subs orde rand ping prod
michaelis_menten_irrevers[204]   0    0    0    0    1    0    0    0    0
michaelis_menten_reversib   0 [110]   0    0    0    0    0    0    0   76
competitive_inhibition      0    0 [133]  59   18    0    0    0    0    0
uncompetitive_inhibition    0    0   66 [126]  13    0    0    0    0    0
mixed_inhibition            0    0   61   52 [ 81]   0    0    0    0    0
substrate_inhibition        3    0    0    0    0 [205]   0    0    0    0
ordered_bi_bi               0    0    0    0    0    0 [ 79]  68   52    0
random_bi_bi                0    0    0    0    0    0   45 [ 96]  49    0
ping_pong                   0    0    0    0    0    0   37   53 [117]   0
product_inhibition          0   34    0    0    0    0    0    0    0 [162]
```

### Auxiliary Task Performance

| Metric | Start | End | Reduction |
|--------|-------|-----|-----------|
| Kinetic Params MSE | 1.69 | 0.12 | 93% |
| Pattern MSE | 9.91 | 1.72 | 83% |

### Key Findings

**The auxiliary tasks were learned well but didn't improve classification.**

1. **Kinetic parameter prediction**: Model learned to predict v0 and Km estimates accurately (MSE dropped 93%)

2. **Pattern prediction**: Model learned parameter change patterns reasonably well (MSE dropped 83%)

3. **But classification stayed flat**: Same confusion patterns as v2
   - MM_rev → product_inhib: 76 errors (was 72)
   - Inhibition triad still confused
   - Bi-substrate triad still confused

### Why Didn't Multi-Task Help?

1. **Auxiliary targets may be too easy**: The model can predict kinetic params without learning the deep discriminative features

2. **Indirect relationship**: Predicting v0/Km doesn't directly require understanding *why* they change differently for each mechanism

3. **Pattern targets too noisy**: The computed slopes have high variance, may not be reliable supervision

4. **Gradient competition**: Auxiliary losses may compete with classification gradients rather than help

### Training Dynamics

```
Best accuracy: 65.9% at epoch 91
Final accuracy: 65.6% at epoch 100
Plateaued at 65-66% for last 30 epochs
```

---

## Remaining Issues

### 1. Inhibition Triad (competitive/uncompetitive/mixed)
- Still significant cross-confusion (v3: 63%, 62%, 42%)
- Multi-task learning didn't help distinguish Km_app vs Vmax_app changes
- **Tried**: Auxiliary kinetic parameter prediction → didn't help
- **Potential fixes**:
  - Explicit Lineweaver-Burk plot features (1/v vs 1/[S] slopes)
  - Hierarchical classification (first detect "inhibition", then subtype)
  - Contrastive learning between inhibition types

### 2. Bi-substrate Triad (ordered/random/ping_pong)
- Improved in v2 (ordered: 3% → 38%), stable in v3 (40%)
- Ping-pong improved significantly in v3 (46% → 57%)
- Still ~equal confusion between all three
- **Potential fixes**:
  - Initial rate analysis at varying [A]/[B] ratios
  - Product inhibition patterns differ between mechanisms
  - More extreme concentration ratios in data generation

### 3. MM_reversible ↔ Product_inhibition
- 76 errors from MM_rev → product_inhib (slightly worse than v2's 72)
- Longer simulation time (5x) in v3 didn't help
- Both show product-dependent slowing, but:
  - MM_rev: True equilibrium (forward = reverse rates)
  - Product_inhib: Slowing but never reaches equilibrium
- **Potential fixes**:
  - Explicit equilibrium detection feature (d²S/dt² near zero)
  - Track approach to equilibrium vs asymptotic slowing
  - Add [P] spiking experiments to data generation

---

## Next Steps

1. ~~**Add auxiliary loss**~~: Tried in v3, didn't help classification
2. **Hierarchical classification**: First classify mechanism family, then subtype
3. **Contrastive learning**: Learn embeddings that separate similar mechanisms
4. **Feature engineering**: Lineweaver-Burk slopes, equilibrium detection
5. **More targeted data**: Experiments specifically designed to distinguish confused pairs
6. **Increase model capacity**: d_model 128 → 256 (may help with subtle patterns)
7. **Ensemble methods**: Train separate specialists for each confusion cluster

---

## Model Architecture

### v2: MultiConditionClassifier

```
MultiConditionClassifier:
├── TrajectoryEncoder (per trajectory)
│   ├── Input projection (5 features → d_model)
│   ├── Positional encoding
│   ├── 1D Conv layers (local patterns)
│   ├── Transformer encoder (global patterns)
│   └── CLS token pooling
├── ConditionEncoder
│   └── MLP: 8 condition features → d_model
├── DerivedFeatureEncoder
│   └── MLP: 8 derived features → d_model
├── Combine layer
│   └── Linear: 3*d_model → d_model
├── Cross-attention (across conditions)
│   └── TransformerEncoder (3 layers)
├── PairwiseComparisonModule
│   ├── Pairwise difference encoding
│   ├── Condition difference encoding
│   └── Self-attention over pairs
├── Attention pooling
└── Classification head
    └── MLP: 2*d_model → 10 mechanisms
```

### v3: MultiTaskClassifier (extends v2)

```
MultiTaskClassifier:
├── [Same as MultiConditionClassifier above]
├── KineticParamHead (auxiliary)
│   └── MLP: d_model → 2 (Vmax_app, Km_app per condition)
├── PatternHead (auxiliary)
│   └── MLP: d_model → 4 (dVmax/dI, dKm/dI, dVmax/dS, dKm/dS)
└── Classification head
    └── MLP: 2*d_model → 10 mechanisms

Loss = CrossEntropy + aux_weight * (KineticMSE + PatternMSE)
aux_weight: 0.5 → 0.1 (curriculum decay over training)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_traj_layers | 2 |
| n_cross_layers | 3 |
| dropout | 0.1 |
| learning_rate | 1e-4 |
| batch_size | 32 |
| max_conditions | 20 |
| n_timepoints | 20 |

---

## Real Data Evaluation

### Dataset: SLAC Laccase (EnzymeML)

**Source**: [EnzymeML SLAC GitHub](https://github.com/EnzymeML/slac_modeling)

**Description**: Small laccase (SLAC) catalyzed oxidation of ABTS substrate, measured across 5 temperatures (25-45°C) with 10 different substrate concentrations.

**Known mechanism**: Michaelis-Menten irreversible

### Data Characteristics

| Property | Value |
|----------|-------|
| Enzyme | Laccase (SLAC) |
| Substrate | ABTS |
| Temperatures | 25, 30, 35, 40, 45°C |
| [S] levels | 10 (0.01 - 8 mM) |
| Time points | 11 per trace |
| Time range | 0 - 900 s |
| Measurement | Absorbance at 420 nm |

### Evaluation Results (v3 Model)

| Metric | Value |
|--------|-------|
| Expected mechanism | `michaelis_menten_irreversible` |
| Predicted mechanism | `michaelis_menten_reversible` |
| Confidence | 66.9% |
| MM_irreversible probability | 24.6% |
| **Result** | **INCORRECT** (but correct family) |

### Full Probability Distribution

```
michaelis_menten_reversible   :  66.9%  <-- PREDICTED
michaelis_menten_irreversible :  24.6%  <-- EXPECTED
substrate_inhibition          :   3.0%
random_bi_bi                  :   2.8%
ordered_bi_bi                 :   1.1%
mixed_inhibition              :   0.6%
competitive_inhibition        :   0.4%
ping_pong                     :   0.3%
uncompetitive_inhibition      :   0.2%
product_inhibition            :   0.0%
```

### Analysis

**Positive findings:**

1. **Model generalizes to real data**: No crashes, produces sensible probability distribution
2. **Correct mechanism family**: Top 2 predictions are both Michaelis-Menten (91.5% combined)
3. **Reasonable confidence**: Not overconfident on wrong answer

**Likely causes of misclassification:**

1. **Preprocessing mismatch**:
   - Real data: absorbance values (0.03 - 3.5 AU)
   - Training data: simulated concentrations (normalized differently)
   - Absorbance-to-concentration conversion may introduce artifacts

2. **Condition structure mismatch**:
   - SLAC: 10 [S] × 5 T = 50 conditions (only [S] and T varied)
   - Training: 20 conditions with [S] × [I] grids
   - Model expects inhibitor variation for MM discrimination

3. **MM_rev vs MM_irrev confusion**:
   - This is a known confusion pair in our model
   - Real laccase kinetics may show equilibrium-like slowing at high conversion
   - Product (oxidized ABTS) may have some inhibitory effect

4. **No inhibitor data**:
   - Model relies heavily on [I] variation to distinguish mechanisms
   - SLAC dataset has no inhibitor experiments

### Implications for Model Improvement

1. **Domain adaptation needed**: Real assay data (absorbance) differs from simulated concentration data

2. **Preprocessing calibration**: Need to match normalization statistics between real and synthetic data

3. **Broader condition coverage**: Training should include temperature-only variation experiments

4. **Mechanism-specific signatures**: MM_irrev vs MM_rev distinction needs features beyond [I] variation:
   - Equilibrium approach behavior
   - Complete substrate depletion patterns
   - Temperature dependence of Keq

### Future Real Data Sources

| Dataset | Source | Mechanism | Status |
|---------|--------|-----------|--------|
| SLAC Laccase | EnzymeML/GitHub | MM irreversible | ✓ Downloaded |
| PGK kinetics | BRENDA/SABIO-RK | Ordered bi-bi | Needs sourcing |
| PFK kinetics | BRENDA/SABIO-RK | Substrate inhibition | Needs sourcing |
| Inhibition data | BRENDA/SABIO-RK | Competitive/uncompetitive | Needs sourcing |

### Scripts

```bash
# Parse SLAC data
python scripts/parse_slac_data.py

# Evaluate on real data
python scripts/evaluate_real_data.py
```

---

## Classical Baseline Comparison

### The Real Baseline

Biochemists don't use ML for mechanism classification. They:
1. Fit each candidate mechanism to their data (nonlinear least squares)
2. Compare residuals + information criteria (AIC, BIC)
3. Use domain knowledge to rule out implausible mechanisms

To claim any kind of "state of the art", TACTIC must beat this baseline.

### Implementation

We implemented classical AIC/BIC model selection:
- Fit all 10 mechanisms to each sample using scipy.optimize
- Parameters fit globally across all conditions (same enzyme)
- Multi-start optimization (3 restarts) to avoid local minima
- AIC = n·ln(SS/n) + 2k, BIC = n·ln(SS/n) + k·ln(n)

### Running the Comparison

```bash
# 1. Generate fixed test set (for reproducibility)
python scripts/generate_test_set.py --n-samples 100 --output data/test_comparison.pt

# 2. Quick comparison (20 samples/mechanism, ~10 min)
python scripts/compare_methods.py --n-samples 20

# 3. Full comparison with saved test set (~30 min)
python scripts/compare_methods.py --load-test-set data/test_comparison.pt --save-results

# 4. Run classical baseline only
python scripts/classical_baseline.py --n-samples 50

# 5. Run TACTIC only (skip classical)
python scripts/compare_methods.py --n-samples 50 --skip-classical
```

### Expected Results

| Method | Expected Accuracy | Notes |
|--------|-------------------|-------|
| Random | 10% | 1/10 classes |
| Classical (AIC) | 40-60%? | Model complexity penalized |
| TACTIC | 66% | Our multi-task model |

### What This Tells Us

**If TACTIC > Classical:**
- ML adds value beyond classical fitting
- Speed advantage is bonus (100-1000x faster)
- Paper claim: "ML outperforms traditional model selection"

**If TACTIC ≈ Classical:**
- ML is faster but not more accurate
- Contribution is speed/automation
- Paper claim: "ML enables rapid mechanism screening at comparable accuracy"

**If TACTIC < Classical:**
- Need to rethink the approach
- Classical methods still win for accuracy
- Focus on interpretability/speed instead

### Scripts

- `scripts/classical_baseline.py` - Classical AIC/BIC model selection
- `scripts/compare_methods.py` - Head-to-head comparison
- `scripts/generate_test_set.py` - Generate reproducible test sets
