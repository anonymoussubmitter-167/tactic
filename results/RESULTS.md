# TACTIC-Kinetics Training Results

All training results for the enzyme mechanism classification model.

---

## Results Summary

| Version | Approach | Accuracy | Date |
|---------|----------|----------|------|
| v0 | Single-curve training | 22% | - |
| v1 | Multi-condition (5 conditions) | 57% | - |
| v2 | Multi-condition + improved strategies | 65% | 2026-01-19 |

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

## Remaining Issues

### 1. Inhibition Triad (competitive/uncompetitive/mixed)
- Still significant cross-confusion
- Model struggles to distinguish Km_app vs Vmax_app changes
- **Potential fix**: Auxiliary loss for kinetic parameter prediction

### 2. Bi-substrate Triad (ordered/random/ping_pong)
- Improved dramatically (ordered: 3% → 38%)
- But still ~equal confusion between all three
- The intersection pattern distinction not fully captured
- **Potential fix**: More extreme [A]/[B] ratios, longer training

### 3. MM_reversible ↔ Product_inhibition
- 72 errors from MM_rev → product_inhib
- Both involve product-dependent kinetics
- **Potential fix**: Longer simulation times to see equilibrium approach

---

## Next Steps

1. **Increase model capacity**: d_model 128 → 256
2. **Add auxiliary loss**: Predict Km_app/Vmax_app to force learning discriminative features
3. **More data**: 2000+ samples per mechanism
4. **Label smoothing**: Prevent overconfidence on wrong classes

---

## Model Architecture

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
