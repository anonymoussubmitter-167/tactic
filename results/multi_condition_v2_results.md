# Multi-Condition Training Results v2

**Date:** 2026-01-19
**Model:** MultiConditionClassifier with PairwiseComparison
**Best Validation Accuracy:** 65.0%

## Summary

| Metric | Value |
|--------|-------|
| Previous accuracy (v1) | 57% |
| Current accuracy (v2) | 65% |
| Improvement | +8% |
| Epochs trained | 100 |
| Converged at | ~epoch 60 |

## Per-Mechanism Accuracy

| Mechanism | v1 Accuracy | v2 Accuracy | Change |
|-----------|-------------|-------------|--------|
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

## Confusion Matrix

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

## Most Confused Pairs

| True Mechanism | Predicted As | Count |
|----------------|--------------|-------|
| uncompetitive_inhibition | competitive_inhibition | 85 |
| ordered_bi_bi | random_bi_bi | 75 |
| mixed_inhibition | competitive_inhibition | 74 |
| michaelis_menten_reversible | product_inhibition | 72 |
| ping_pong | random_bi_bi | 70 |
| competitive_inhibition | uncompetitive_inhibition | 50 |
| ordered_bi_bi | ping_pong | 49 |
| random_bi_bi | ordered_bi_bi | 47 |
| mixed_inhibition | uncompetitive_inhibition | 46 |
| random_bi_bi | ping_pong | 46 |

## Training Dynamics

```
Epochs 51-75:  63.5% avg
Epochs 76-100: 64.3% avg
Change: +0.7% (plateaued)
```

Model converged around epoch 60-65, oscillating between 63-65% for final 50 epochs.

## Changes from v1

1. **Improved bi-substrate condition variation**
   - 4 slices through [A]-[B] space (20 conditions total)
   - Fix [B] at low/high, vary [A] widely
   - Fix [A] at low/high, vary [B] widely
   - Reveals intersection pattern for mechanism discrimination

2. **Full [I] × [S] grid for inhibition**
   - 4 [I] levels × 5 [S] levels = 20 conditions
   - Enables measurement of Km_app and Vmax_app changes with [I]

3. **Track multiple species**
   - Now track both S and P (not just S)
   - 5 trajectory features: t, S, P, dS/dt, dP/dt

4. **Derived kinetic features**
   - v0 (initial rate)
   - t_half (time to 50% conversion)
   - rate_ratio (late/early rate)
   - final_conversion
   - P_final, mass_balance_error
   - exponential_residual, acceleration

5. **Pairwise comparison module**
   - Explicit comparison between condition pairs
   - Self-attention over pairs
   - Combined with cross-attention pooled features

## Remaining Issues

### Inhibition Triad (competitive/uncompetitive/mixed)
- Still significant cross-confusion
- Model struggles to distinguish Km_app vs Vmax_app changes
- Need: auxiliary loss for kinetic parameter prediction

### Bi-substrate Triad (ordered/random/ping_pong)
- Improved dramatically (ordered: 3% → 38%)
- But still ~equal confusion between all three
- The intersection pattern distinction not fully captured

### MM_reversible ↔ Product_inhibition
- 72 errors from MM_rev → product_inhib
- Both involve product-dependent kinetics
- Need: longer simulation times to see equilibrium approach

## Next Steps

1. **Increase model capacity**: d_model 128 → 256
2. **Add auxiliary loss**: predict Km_app/Vmax_app to force learning discriminative features
3. **More data**: 2000+ samples per mechanism
4. **Label smoothing**: prevent overconfidence on wrong classes
