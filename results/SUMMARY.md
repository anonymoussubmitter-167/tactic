# TACTIC Results Summary

TACTIC (Transformer Architecture for Classifying Thermodynamic and Inhibition Characteristics) is a deep learning approach for classifying enzyme mechanisms from kinetic time-course data. This document summarizes our benchmark results comparing TACTIC against classical model selection (AIC-based curve fitting).

## Overall Accuracy (1000 test samples)

| Method | Accuracy | vs Classical |
|--------|----------|--------------|
| Random baseline | 10.0% | - |
| Classical (AIC) | 38.6% | baseline |
| **v1** (basic multi-cond) | 10.3% | -28.3% |
| **v2** (improved) | 56.1% | +17.5% |
| **v3** (multi-task) | **62.0%** | **+23.4%** |

**Note**: v1 results indicate the model collapsed to predicting a single class (MM_reversible), likely due to training issues with the basic architecture. The meaningful comparison is v2/v3 vs classical.

## Version Progression

| Transition | Gain | Key Innovation |
|------------|------|----------------|
| v1 → v2 | +45.8% | Derived features + pairwise comparison + architecture fixes |
| v2 → v3 | +5.9% | Multi-task auxiliary heads |

**Interpretation**: The largest gains come from derived kinetic features (v0, Km estimates, curvature) and improved architecture. Multi-task learning provides additional regularization.

---

## Experiment Results

### Experiment 1: Confidence-Accuracy Calibration

**Question**: When TACTIC reports high confidence, can we trust it?

| Version | Overall Acc | Acc at >90% conf | ECE | Mean Conf |
|---------|-------------|------------------|-----|-----------|
| v1 | 10.3% | N/A (no high conf) | 0.41 | 0.51 |
| v2 | 56.1% | 82.8% | ~0.15 | 0.73 |
| v3 | 62.0% | **98.0%** | **0.064** | 0.66 |

**v3 Confidence Breakdown:**
| Confidence | Accuracy | N Samples |
|------------|----------|-----------|
| 0.9 - 1.0 | **98.0%** | 307 |
| 0.8 - 0.9 | 74.0% | 73 |
| 0.6 - 0.8 | 60.2% | 171 |
| 0.4 - 0.6 | 39.0% | 305 |
| 0.2 - 0.4 | 29.9% | 144 |

**Key finding**: v3 is well-calibrated (ECE=0.064). When confident (>90%), it's correct 98% of the time. v1 shows no discrimination (all predictions ~51% confidence).

---

### Experiment 2: Condition Ablation

**Question**: How many experimental conditions do researchers need?

| N Conditions | v1 | v2 | v3 |
|--------------|-----|-----|-----|
| 1 | 9.8% | 13.8% | 12.8% |
| 2 | 9.8% | 31.0% | 22.3% |
| 3 | 10.4% | 32.8% | 29.5% |
| 5 | 10.2% | 36.6% | 33.1% |
| 7 | 10.2% | 52.6% | 49.9% |
| 10 | 10.4% | 54.4% | 57.5% |
| 15 | 10.2% | 56.4% | 59.8% |
| 20 | 10.4% | 57.4% | 62.0% |

**Key finding**: Single-curve experiments are fundamentally insufficient (~10-14% accuracy). The sweet spot is 7-10 conditions. Diminishing returns after 10 conditions.

---

### Experiment 3: Noise Robustness

**Question**: How does TACTIC perform with noisy experimental data?

| Noise Level | v1 | v2 | v3 |
|-------------|-----|-----|-----|
| 0% | 10.0% | 59.4% | 62.0% |
| 5% | 10.0% | 57.8% | 62.4% |
| 10% | 10.0% | 55.2% | 60.2% |
| 20% | 10.0% | 55.0% | 59.0% |
| 30% | 10.0% | 52.4% | 57.9% |
| **Degradation** | 0% | 7.0% | **4.1%** |

**Key finding**: v3 is very robust - only 4.1% accuracy degradation at 30% measurement noise. Suitable for real experimental data with typical 5-15% measurement error.

---

### Experiment 4: Family-Level Accuracy

**Question**: Even when exact mechanism is wrong, does TACTIC get the family right?

| Version | 10-Class Acc | 5-Family Acc | Improvement |
|---------|--------------|--------------|-------------|
| v1 | 9.9% | 20.0% | +10.1% |
| v2 | 61.9% | 99.2% | +37.3% |
| v3 | 62.0% | **99.6%** | **+37.6%** |

**Families:**
- **simple**: Michaelis-Menten irreversible
- **reversible**: MM reversible, product inhibition
- **inhibited**: competitive, uncompetitive, mixed inhibition
- **substrate_regulated**: substrate inhibition
- **bisubstrate**: ordered bi-bi, random bi-bi, ping-pong

**v3 Per-Family Accuracy:**
| Family | Accuracy |
|--------|----------|
| simple | 99% |
| reversible | 100% |
| inhibited | 100% |
| substrate_regulated | 97% |
| bisubstrate | 100% |

**Key finding**: v3 almost never makes catastrophic errors. 99.6% family accuracy means errors are within biochemically similar subtypes. Only 4 between-family errors vs 376 within-family errors.

---

### Experiment 5: Identifiability Analysis

**Question**: Which mechanism pairs are theoretically confusable?

**v3 Most Confused Pairs:**
| Confused Pair | Confusion Rate |
|---------------|----------------|
| competitive ↔ uncompetitive | 32.5% |
| ordered_bi_bi ↔ ping_pong | 32.5% |
| MM_reversible ↔ product_inhibition | 32.0% |
| ordered_bi_bi ↔ random_bi_bi | 31.0% |
| competitive ↔ mixed | 24.5% |

**Interpretation**: These confusions reflect fundamental biochemical ambiguities:
- **Inhibition types** produce similar Lineweaver-Burk patterns
- **Bisubstrate mechanisms** have overlapping initial velocity patterns
- **Reversible/product inhibition** both show product accumulation effects

---

### Experiment 6: Error Correlation Analysis

**Question**: Do TACTIC and classical methods make the same mistakes?

| Metric | v1 | v2 | v3 |
|--------|-----|-----|-----|
| TACTIC accuracy | 10.1% | 56.1% | 62.0% |
| Classical accuracy | 39.4% | 39.4% | 38.8% |
| Error correlation | 0.21 | **0.07** | 0.21 |
| Ensemble upper bound | 42.4% | 71.7% | **71.8%** |

**v3 Contingency Table:**
|  | Classical ✓ | Classical ✗ |
|--|-------------|-------------|
| **TACTIC ✓** | 290 (29.0%) | 330 (33.0%) |
| **TACTIC ✗** | 98 (9.8%) | 282 (28.2%) |

**Key finding**: Low error correlation (0.07-0.21) indicates complementary strengths. An ensemble could reach 71.8% accuracy - 10% above either method alone. TACTIC succeeds where classical fails 33% of the time.

---

### Experiment 7: Literature Cases & Speed

**Well-characterized enzymes from literature** (ground truth mechanisms):
1. Alcohol dehydrogenase - ordered_bi_bi
2. Lactate dehydrogenase - ordered_bi_bi
3. Hexokinase - random_bi_bi
4. Aspartate aminotransferase - ping_pong
5. Chymotrypsin + benzamidine - competitive_inhibition
6. Acetylcholinesterase + edrophonium - competitive_inhibition
7. Alkaline phosphatase + L-phenylalanine - uncompetitive_inhibition
8. Xanthine oxidase - substrate_inhibition
9. Phosphofructokinase - substrate_inhibition
10. Fumarase - michaelis_menten_reversible

**Speed Comparison (typical):**
- TACTIC: ~2-5 ms per sample
- Classical: ~1-2 seconds per sample
- **Speedup: ~134x faster**

**Speed Comparison (literature cases - single sample inference):**
| Version | TACTIC (ms) | Classical (s) | Speedup |
|---------|-------------|---------------|---------|
| v1 | 5.0 | 5.87 | 1175x |
| v2 | 16.8 | 5.51 | 328x |
| v3 | 19.0 | 5.89 | 310x |

**Note**: Single-sample inference on complex literature cases shows 310-1175x speedup. Batch processing typically achieves ~134x speedup.

---

## Key Findings Summary

1. **TACTIC v3 beats classical by +23.4%** (62.0% vs 38.6%)

2. **Single-curve experiments are fundamentally limited** - ~10-14% accuracy confirms that mechanism discrimination requires observing kinetic responses across conditions

3. **7-10 conditions provide optimal accuracy** - diminishing returns beyond this range

4. **High confidence predictions are reliable** - 98% accurate when confidence >90% (ECE=0.064)

5. **Family-level classification is near-perfect** - 99.6% accuracy, errors are within biochemically similar subtypes

6. **Robust to experimental noise** - only 4% degradation at 30% noise

7. **Complementary to classical methods** - low error correlation (0.07-0.21), ensemble could reach 72%

8. **~134x faster than classical fitting** (up to 1175x for single-sample inference) - enables high-throughput screening

9. **Confusion patterns match biochemical theory** - validates learned representations

---

## Experiments Status

| # | Experiment | v1 | v2 | v3 |
|---|------------|-----|-----|-----|
| 1 | Confidence Analysis | ✓ | ✓ | ✓ |
| 2 | Condition Ablation | ✓ | ✓ | ✓ |
| 3 | Noise Robustness | ✓ | ✓ | ✓ |
| 4 | Family Accuracy | ✓ | ✓ | ✓ |
| 5 | Identifiability | ✓ | ✓ | ✓ |
| 6 | Error Correlation | ✓ | ✓ | ✓ |
| 7 | Literature Cases | ✓ | ✓ | ✓ |

All experiments complete for all versions.
