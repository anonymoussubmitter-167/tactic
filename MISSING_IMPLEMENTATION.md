# TACTIC-Kinetics: Missing Implementation Items

This document compares the original project plan to the current implementation and lists all missing components.

---

## 1. Core Thesis Implementation Status

| Plan Component | Status | Notes |
|----------------|--------|-------|
| Gibbs energy parameterization | ✓ Implemented | State energies + barrier energies |
| Thermodynamic consistency by construction | ✓ Implemented | Eyring equation, Haldane relationships |
| Mechanism discrimination via energy profiles | ⚠️ Partial | Classifier exists but uses latent h, not energy profiles directly |
| Transferable priors from eQuilibrator | ⚠️ Partial | Using TECRDB K_prime, but not full component contribution |

---

## 2. Data Gaps

### 2.1 Thermodynamic Grounding Data

| Data Source | Plan Requirement | Current Status | Missing |
|-------------|------------------|----------------|---------|
| eQuilibrator Compounds | 1.3GB SQLite for ~500k compounds | Not downloaded | **Full compound database for arbitrary reaction queries** |
| eQuilibrator API | Query ΔG° for any reaction | API wrapper exists but untested | **Tested integration with real queries** |
| BRENDA Flat File | Derive empirical kcat, Km, Ki distributions | Only API script | **Actual BRENDA data extraction** |
| BRENDA kcat→ΔG‡ | Convert turnover numbers to activation energies | Hardcoded literature values | **Real BRENDA kcat distribution analysis** |
| BRENDA Km→ΔG_bind | Convert Km to binding energies | Not implemented | **Km to binding energy conversion** |
| SABIO-RK | 71,000 kinetic entries | Only API script | **Bulk data download and parsing** |

### 2.2 Real Kinetic Datasets

| Dataset | Plan Requirement | Current Status | Missing |
|---------|------------------|----------------|---------|
| EnzymeML SLAC-ABTS | Full progress curves at T=25-45°C | Files downloaded | **Validated data loading and parsing** |
| EnzymeML SLAC | Cross-temperature validation benchmark | OMEX files exist | **Benchmark pipeline using this data** |
| SulfoSys PGK | Time courses with [substrate] variation | Excel file downloaded | **Excel parser and data loader** |
| PFK Kinetics | Steady-state + progress curves | Excel file downloaded | **Excel parser and data loader** |

---

## 3. Energy Landscape Parameterization Gaps

### 3.1 State Energies (Section 3.1 of Plan)

| State | Plan Definition | Implementation | Missing |
|-------|-----------------|----------------|---------|
| G_ES | Enzyme-substrate complex | ✓ Implemented | - |
| G_EP | Enzyme-product complex | ✓ Implemented | - |
| G_EI | Enzyme-inhibitor complex | Defined in templates | **Not used in ODE simulation** |
| G_ESI | Ternary complex (substrate + inhibitor) | Not implemented | **Ternary complex states** |
| G_EIS | Ternary complex (inhibitor + substrate) | Not implemented | **Ternary complex states** |

### 3.2 Transition State Energies

| Barrier | Plan Definition | Implementation | Missing |
|---------|-----------------|----------------|---------|
| G‡_bind | Substrate binding transition state | ✓ Via BRENDA "binding" prior | - |
| G‡_cat | Catalytic transition state | ✓ Via BRENDA "catalysis" prior | - |
| G‡_release | Product release transition state | ✓ Via BRENDA "release" prior | - |
| G‡_inhibitor | Inhibitor binding barrier | ✓ Via BRENDA "inhibitor_binding" prior | **Not connected to simulation** |

### 3.3 Derived Kinetic Parameters

| Parameter | Plan Formula | Implementation | Missing |
|-----------|--------------|----------------|---------|
| k_cat | (k_B T / h) · exp(-ΔG‡_cat / RT) | ✓ Eyring equation | - |
| K_m | exp(ΔG_bind / RT) · [1 + ...] | Simplified formula | **Full derivation with rate ratios** |
| K_eq | exp(-ΔG°_rxn / RT) | ✓ Implemented | - |
| K_i | Inhibitor dissociation constant | Not derived | **K_i from G_EI** |

---

## 4. Mechanism Template Gaps

### 4.1 Defined Mechanisms vs Plan

| Mechanism | Plan Section | Implementation | Simulation Works? |
|-----------|--------------|----------------|-------------------|
| Irreversible Michaelis-Menten | 3 nodes, 2 edges | ✓ 3 states, 2 transitions | ✓ Yes (Euler) |
| Reversible Michaelis-Menten | Implied | ✓ 4 states, 3 transitions | ⚠️ Untested |
| Competitive Inhibition | 4 nodes, 3 edges | ✓ 4 states, 3 transitions | **No - missing I concentration** |
| Uncompetitive Inhibition | Implied | ✓ 4 states, 3 transitions | **No - missing I concentration** |
| Mixed Inhibition | Implied | ✓ 5 states, 5 transitions | **No - missing I concentration** |
| Substrate Inhibition | Mentioned | ✓ 4 states, 3 transitions | **No - missing excess S effect** |
| Product Inhibition | Mentioned | ✓ 4 states, 4 transitions | **No - missing P concentration** |
| Ordered Bi-Bi | 6 nodes, 5 edges | ✓ 6 states, 5 transitions | **No - only 1 substrate in ODE** |
| Random Bi-Bi | 7 nodes, 8 edges | ✓ 6 states, 6 transitions | **No - only 1 substrate in ODE** |
| Ping-Pong | Implied | ✓ 8 states, 6 transitions | **No - only 1 substrate in ODE** |

### 4.2 Missing Mechanism Features

```
MISSING: Transition type annotations in mechanism templates

Current:
  Transition(name="k1", from_state="E_S", to_state="ES", is_reversible=True)

Required:
  Transition(name="k1", from_state="E_S", to_state="ES",
             is_reversible=True, transition_type="binding")  # <-- MISSING
```

```
MISSING: Multi-substrate ODE systems

Current ODE (SimpleMichaelisMentenODE):
  dS/dt = -v
  dP/dt = +v

Required for Bi-Bi:
  dA/dt = -v_bind_A
  dB/dt = -v_bind_B
  dP/dt = +v_release_P
  dQ/dt = +v_release_Q
```

```
MISSING: Inhibitor concentration in simulation

Current: No inhibitor species tracked
Required:
  - [I] as input condition
  - EI, ESI states in ODE
  - Competitive: ES + I ⇌ EI (blocks substrate)
  - Uncompetitive: ES + I ⇌ ESI (blocks turnover)
```

---

## 5. Neural Architecture Gaps

### 5.1 Energy Landscape Encoder (Section 3.3.A)

| Component | Plan | Implementation | Missing |
|-----------|------|----------------|---------|
| Input: sparse observations | (t_i, y_i) + conditions | ✓ times, values, conditions, mask | - |
| Transformer encoder | With learned time embeddings | ✓ Sinusoidal embeddings | - |
| Condition embedding | T, pH, [S]_0, [I] | ✓ 4 conditions (T, pH, S0, E0) | **[I] inhibitor concentration** |
| Output: latent h | h ∈ ℝ^d | ✓ h ∈ ℝ^256 | - |

### 5.2 Mechanism-Specific Energy Decoders (Section 3.3.B)

| Component | Plan | Implementation | Missing |
|-----------|------|----------------|---------|
| Per-mechanism MLP | f_m: h → {G_state, G‡} | ✓ MechanismSpecificDecoder | - |
| Energies in kJ/mol | Unconstrained reals | ✓ No positivity constraints | - |
| Rate constant computation | Via Eyring equation | ✓ EnergyToRateConverter | - |
| **Energy profile classification** | Classify on energy profiles | Uses latent h only | **Classifier should use decoded energies** |

### 5.3 Mechanism Classifier (Section 3.3.C)

| Component | Plan | Implementation | Missing |
|-----------|------|----------------|---------|
| p(m \| h) = softmax(W_m · h) | Classify from latent | ✓ MechanismClassifier | - |
| Learn mechanism signatures | From observation patterns | ✓ Trained on mechanism_idx | - |
| **Use energy landscape features** | "Classification operates on energy profiles" | Not implemented | **Classifier uses h, not energies** |

### 5.4 Forward Simulator (Section 3.3.D)

| Component | Plan | Implementation | Missing |
|-----------|------|----------------|---------|
| energies → rate constants | Via Eyring | ✓ Implemented | - |
| Build ODE from mechanism | General mechanism ODE | Only MM ODE | **General MechanismODE not used** |
| Differentiable solve | odeint | ✓ torchdiffeq (+ Euler fallback) | - |
| **Multi-substrate** | "build_ode(mechanism, ...)" | Only 1-substrate | **Bi-substrate ODE systems** |

---

## 6. Training Objective Gaps

### 6.1 Loss Functions (Section 3.4)

| Loss | Plan Formula | Implementation | Missing |
|------|--------------|----------------|---------|
| L_traj | Σ_i \|\|y_i - ŷ(t_i)\|\|² | ✓ TACTICLoss.trajectory_loss | **Not used in train_full.py** |
| L_thermo | \|\|ΔG°_pred - ΔG°_eQuilibrator\|\|² | ✓ TACTICLoss.thermodynamic_loss | **No eQuilibrator supervision data** |
| L_mech | -Σ_m y_m log p(m \| h) | ✓ CrossEntropyLoss | - |
| L_prior | Σ \|\|G - G°_CC\|\|² / σ² | ✓ TACTICLoss.prior_loss | **Not using real CC estimates** |

### 6.2 Training Pipeline Gaps

```
MISSING: Trajectory reconstruction training

Current train_full.py:
  loss = criterion(outputs["mechanism_logits"], labels)  # Classification only

Plan requires:
  losses = loss_fn(predictions, targets)
  # where targets includes:
  #   - trajectory: ground truth time series
  #   - mechanism_labels: mechanism index
  #   - known_dg: eQuilibrator ΔG° values
```

```
MISSING: Component Contribution supervision

Plan: "L_prior = Σ ||G_state - G°_CC||² / σ²_CC"

Current: Uses hardcoded prior means
Required: Query eQuilibrator for reaction-specific ΔG°
```

---

## 7. Synthetic Data Generation Gaps

### 7.1 Parameter Sampling (Section 4.3)

| Sampling Step | Plan | Implementation | Missing |
|---------------|------|----------------|---------|
| ΔG° from component contribution | eQuilibrator distributions | ✓ TECRDB K_prime → ΔG° | **Per-reaction CC estimates** |
| ΔG‡ from BRENDA | Empirical activation energies | Hardcoded literature values | **Real BRENDA extraction** |
| Thermodynamic validity | "All synthetic data valid by construction" | ✓ Energy-based sampling | - |
| Heteroscedastic noise | "Calibrated to real assay variability" | Fixed noise_std=0.02 | **Noise model from real data** |

### 7.2 Mechanism Coverage

| Family | Plan Requirement | Implementation | Notes |
|--------|------------------|----------------|-------|
| MM irreversible | ✓ | ✓ | Works |
| MM reversible | ✓ | ✓ | Works |
| Competitive inhibition | ✓ | ✓ | **Simulation broken** |
| Uncompetitive inhibition | ✓ | ✓ | **Simulation broken** |
| Mixed inhibition | ✓ | ✓ | **Simulation broken** |
| Product inhibition | ✓ | ✓ | **Simulation broken** |
| Ordered bi-substrate | ✓ | ✓ | **Simulation broken** |
| Random bi-substrate | ✓ | ✓ | **Simulation broken** |
| Substrate inhibition | ✓ | ✓ | **Simulation broken** |
| Ping-Pong | Implied | ✓ | **Simulation broken** |

---

## 8. Experiments Gaps

### 8.1 Experiment 1: Mechanism Discrimination

| Requirement | Status | Missing |
|-------------|--------|---------|
| Synthetic curves from different mechanisms | ✓ Generator exists | - |
| Similar steady-state behavior | Not controlled | **Generate confusable mechanisms** |
| Compare TACTIC vs AIC/BIC vs LRT | Not implemented | **Baseline implementations** |
| Confusion matrix analysis | ✓ In train_full.py | - |

### 8.2 Experiment 2: Parameter Recovery

| Requirement | Status | Missing |
|-------------|--------|---------|
| Compare kinetic vs thermodynamic parameterization | Not implemented | **Kinetic baseline model** |
| Parameter RMSE | Not computed | **Ground truth comparison** |
| Profile likelihood analysis | Not implemented | **Identifiability analysis** |

### 8.3 Experiment 3: Transferability

| Requirement | Status | Missing |
|-------------|--------|---------|
| Train on enzyme class A, test on class B | Not implemented | **Cross-enzyme evaluation** |
| Pretrained kinetic vs thermodynamic priors | Not implemented | **Prior transfer experiments** |
| Few-shot inference (1, 3, 5 curves) | Not implemented | **Few-shot evaluation** |

### 8.4 Experiment 4: Temperature Robustness

| Requirement | Status | Missing |
|-------------|--------|---------|
| Train at T=25°C, test at T=37°C | Not implemented | **Temperature split evaluation** |
| Eyring temperature dependence | ✓ In rate computation | - |
| Extrapolation beyond training range | Not evaluated | **Out-of-distribution testing** |

### 8.5 Experiment 5: Real Data Case Studies

| Requirement | Status | Missing |
|-------------|--------|---------|
| EnzymeML SLAC cross-temperature | Data exists | **Validation pipeline** |
| PGK/PFK leave-one-condition-out | Data exists | **Excel loader + evaluation** |
| Comparison vs COPASI | Not implemented | **COPASI baseline** |
| Comparison vs Neural ODE | Not implemented | **Neural ODE baseline** |
| Comparison vs SBI in kinetic coords | Not implemented | **SBI baseline** |

---

## 9. Active Experimental Design Gaps

### 9.1 Information-Theoretic Acquisition (Section 6.1)

| Component | Status | Missing |
|-----------|--------|---------|
| Mutual information I(G; y_a \| D) | Not implemented | **Acquisition function** |
| argmax over conditions | Not implemented | **Condition optimizer** |
| Batch acquisition | Not implemented | **Batch selection strategy** |

### 9.2 Thermodynamic-Aware Strategies (Section 6.2)

| Strategy | Status | Missing |
|----------|--------|---------|
| High [S] probes G‡_cat | Not implemented | **Condition-to-energy mapping** |
| Low [S] probes G_ES | Not implemented | **Binding regime detection** |
| Add inhibitor probes G_EI | Not implemented | **Inhibitor titration design** |
| Temperature perturbation for ΔG‡ vs ΔG° | Not implemented | **T-sweep design** |

### 9.3 Experimental Comparison (Section 6.3)

| Baseline | Status | Missing |
|----------|--------|---------|
| Uniform time sampling | Not implemented | **Random baseline** |
| D-optimal design | Not implemented | **Fisher information baseline** |
| Random condition selection | Not implemented | **Random baseline** |
| TACTIC planner | Not implemented | **Active learning loop** |

---

## 10. Code Infrastructure Gaps

### 10.1 Missing Files

| File | Purpose | Status |
|------|---------|--------|
| `data/fairdomhub_loader.py` | Load PGK/PFK Excel files | **Not created** |
| `data/brenda_loader.py` | Extract BRENDA kcat/Km distributions | **Not created** |
| `data/sabio_rk_loader.py` | Load SABIO-RK kinetic data | **Not created** |
| `models/mechanism_ode.py` | General multi-substrate ODE | **Only SimpleMichaelisMentenODE** |
| `evaluation/baselines.py` | COPASI, Neural ODE, SBI baselines | **Not created** |
| `evaluation/benchmarks.py` | Experiments 1-5 implementation | **Not created** |
| `active_learning/acquisition.py` | Experimental design | **Not created** |
| `tests/*.py` | Unit and integration tests | **Not created** |

### 10.2 Missing Functionality in Existing Files

```python
# tactic_kinetics/mechanisms/templates.py
# MISSING: transition_type annotation
class Transition:
    # Current: name, from_state, to_state, is_reversible
    # Missing: transition_type: Literal["binding", "catalysis", "release", "inhibitor_binding"]

# tactic_kinetics/models/ode_simulator.py
# MISSING: General mechanism ODE (not just MM)
class MechanismODE:
    # Exists but not used - only SimpleMichaelisMentenODE is used

# tactic_kinetics/training/synthetic_data.py
# MISSING: Simulation for non-MM mechanisms
def generate_trajectory_for_mechanism(mechanism, energies, conditions):
    # Currently all mechanisms use generate_trajectory_simple() which is MM-only
```

---

## 11. Validation Gaps

### 11.1 Unit Tests

| Module | Test Coverage | Missing Tests |
|--------|---------------|---------------|
| thermodynamic_priors.py | 0% | **EnergyDistribution sampling, TECRDB parsing** |
| templates.py | 0% | **Mechanism graph structure, energy param counts** |
| encoder.py | 0% | **Forward pass shapes, attention masking** |
| ode_simulator.py | 0% | **Rate constant computation, Euler stability** |
| synthetic_data.py | 0% | **Prior-based sampling, trajectory generation** |
| enzymeml_loader.py | 0% | **OMEX parsing, time series extraction** |

### 11.2 Integration Tests

| Test | Status | Missing |
|------|--------|---------|
| End-to-end training | Manual only | **Automated test** |
| Synthetic → Train → Evaluate | Not tested | **Pipeline test** |
| Real data loading → Inference | Not tested | **Real data test** |
| Energy → Rate → ODE → Trajectory | Partially tested | **Full forward test** |

### 11.3 Validation Against Known Values

| Check | Status | Missing |
|-------|--------|---------|
| Eyring equation produces correct rates | Not validated | **Known barrier → known rate** |
| MM simulation matches analytical solution | Not validated | **Analytical comparison** |
| Thermodynamic consistency (Haldane) | Not validated | **K_eq = k_f/k_r check** |
| eQuilibrator values match literature | Not validated | **Spot checks** |

---

## 12. Summary: Critical Path to Complete Implementation

### Must Fix (Blocking)

1. **Multi-substrate ODE simulation** - 6 of 10 mechanisms cannot be simulated
2. **Inhibitor concentration handling** - 4 inhibition mechanisms non-functional
3. **Trajectory reconstruction loss** - Currently classification-only training

### Should Fix (Quality)

4. **Real BRENDA data extraction** - Currently using literature approximations
5. **Transition type annotations** - Needed for proper energy assignment
6. **EnzymeML validation** - Prove system works on real data
7. **FAIRDOMHub loaders** - Enable PGK/PFK benchmarks

### Nice to Have (Completeness)

8. **Active experimental design** - Secondary contribution per plan
9. **All 5 experiments** - Full evaluation suite
10. **Baseline comparisons** - COPASI, Neural ODE, SBI
11. **Unit tests** - Code quality

---

## 13. Estimated Effort

| Category | Items | Effort |
|----------|-------|--------|
| Data gaps | BRENDA extraction, loaders | 2-3 days |
| ODE fixes | Multi-substrate, inhibitor | 2-3 days |
| Training fixes | Full loss function, trajectory | 1 day |
| Validation | EnzymeML, FAIRDOMHub | 1-2 days |
| Experiments | 5 experiments + baselines | 3-5 days |
| Active learning | Acquisition functions | 2-3 days |
| Tests | Unit + integration | 2-3 days |
| **Total** | | **~2-3 weeks** |
