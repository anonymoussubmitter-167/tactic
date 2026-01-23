# TACTIC Results Summary

## Overall Accuracy (1000 test samples)

| Method | Accuracy | vs Classical |
|--------|----------|--------------|
| Random baseline | 10.0% | - |
| Classical (AIC) | 38.6% | baseline |
| **v0** (single-curve) | 10.6% | -28.0% |
| **v1** (basic multi-cond) | 48.6% | +10.0% |
| **v2** (improved) | 61.6% | +23.0% |
| **v3** (multi-task) | **62.0%** | **+23.4%** |

## Version Progression

| Transition | Gain | Key Innovation |
|------------|------|----------------|
| v0 → v1 | +38.0% | Multiple conditions (5 curves vs 1) |
| v1 → v2 | +13.0% | Derived features + pairwise comparison |
| v2 → v3 | +0.4% | Multi-task auxiliary heads |

## Per-Mechanism Accuracy

| Mechanism | v0 | v1 | v2 | v3 | Classical |
|-----------|-----|-----|-----|-----|-----------|
| michaelis_menten_irreversible | 0% | 99% | 97% | **99%** | 52% |
| michaelis_menten_reversible | 0% | 66% | 60% | 54% | **66%** |
| competitive_inhibition | 96% | 52% | 53% | **61%** | 19% |
| uncompetitive_inhibition | 0% | 21% | 59% | **61%** | 24% |
| mixed_inhibition | 0% | 20% | **47%** | 36% | 29% |
| substrate_inhibition | 0% | 95% | **100%** | 97% | 69% |
| ordered_bi_bi | 0% | 9% | **38%** | 32% | 9% |
| random_bi_bi | 0% | **55%** | 39% | 41% | 48% |
| ping_pong | 9% | 24% | 54% | **57%** | 46% |
| product_inhibition | 1% | 45% | 69% | **82%** | 24% |

---

## Experiment Results

### Experiment 1: Confidence-Accuracy Calibration

**Key finding**: When TACTIC is confident, it's correct. ECE = 0.064 (well-calibrated)

| Confidence | Accuracy | N Samples |
|------------|----------|-----------|
| 0.9 - 1.0 | **98.0%** | 307 |
| 0.8 - 0.9 | 74.0% | 73 |
| 0.6 - 0.8 | 60.2% | 171 |
| 0.4 - 0.6 | 39.0% | 305 |
| 0.2 - 0.4 | 29.9% | 144 |

### Experiment 2: Condition Ablation

**Key finding**: Need 7-10 conditions for good accuracy. Diminishing returns after 10.

| N Conditions | Accuracy |
|--------------|----------|
| 1 | 12.8% |
| 2 | 22.3% |
| 3 | 29.5% |
| 5 | 33.1% |
| 7 | 49.9% |
| 10 | 57.5% |
| 15 | 59.8% |
| 20 | 62.0% |

### Experiment 3: Noise Robustness

**Key finding**: Very robust - only 4.1% degradation at 30% noise.

| Noise Level | Accuracy | Degradation |
|-------------|----------|-------------|
| 0% | 62.0% | baseline |
| 5% | 62.4% | +0.4% |
| 10% | 60.2% | -1.8% |
| 20% | 59.0% | -3.0% |
| 30% | 57.9% | -4.1% |

### Experiment 4: Family-Level Accuracy

**Key finding**: 99.6% accuracy when collapsing to 5 mechanism families.

| Level | Accuracy |
|-------|----------|
| 10-class (mechanism) | 62.0% |
| 5-class (family) | **99.6%** |
| Improvement | +37.6% |

**Families**: simple, reversible, inhibited, substrate_regulated, bisubstrate

### Experiment 5: Identifiability Analysis

**Key finding**: Most confusions are within theoretically expected groups.

| Confused Pair | Confusion Rate |
|---------------|----------------|
| competitive ↔ uncompetitive | 32.5% |
| ordered_bi_bi ↔ ping_pong | 32.5% |
| MM_reversible ↔ product_inhibition | 32.0% |
| ordered_bi_bi ↔ random_bi_bi | 31.0% |
| competitive ↔ mixed | 24.5% |

### Experiment 6: Error Correlation Analysis

**Key finding**: TACTIC and classical methods have complementary strengths (low error correlation).

| Metric | Value |
|--------|-------|
| Both correct | 290 (29.0%) |
| TACTIC only correct | 330 (33.0%) |
| Classical only correct | 98 (9.8%) |
| Both wrong | 282 (28.2%) |
| Error correlation | 0.209 (low) |
| Ensemble upper bound | **71.8%** |

**Insight**: Low correlation (0.209) suggests combining methods could reach 71.8% accuracy.

---

## Key Findings

1. **TACTIC beats classical by +23.4%** (62.0% vs 38.6%)
2. **Single-curve is insufficient** - v0 performs at random chance (10.6%)
3. **Multi-condition is critical** - v1 gains +38% from using 5 conditions
4. **Derived features help** - v2 gains +13% from kinetic features
5. **High confidence = reliable** - 98% accurate when conf > 0.9
6. **Family-level near-perfect** - 99.6% at 5-family level
7. **Noise-robust** - only 4% drop at 30% noise
8. **Confusions match theory** - inhibition types and bisubstrate patterns
9. **Complementary to classical** - low error correlation (0.21), ensemble could reach 72%

## Experiments Status

| # | Experiment | Status |
|---|------------|--------|
| 1 | Confidence Analysis | ✓ Complete |
| 2 | Condition Ablation | ✓ Complete |
| 3 | Noise Robustness | ✓ Complete |
| 4 | Family Accuracy | ✓ Complete |
| 5 | Identifiability | ✓ Complete |
| 6 | Error Correlation | ✓ Complete |
| 7 | Literature Cases | Pending |
