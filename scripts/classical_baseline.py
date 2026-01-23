#!/usr/bin/env python
"""
Classical AIC/BIC model selection baseline for enzyme mechanism classification.

This implements what biochemists actually do:
1. Fit each candidate mechanism to the data (nonlinear least squares)
2. Compare residuals + information criteria (AIC, BIC)
3. Select the mechanism with lowest AIC/BIC

This is the real baseline TACTIC needs to beat.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tactic_kinetics.training.multi_condition_generator import (
    MultiConditionSample,
    MultiConditionGenerator,
    load_dataset,
)


# Mechanism parameter specifications
# Each mechanism: (param_names, n_params, requires_inhibitor, requires_two_substrates)
MECHANISM_SPECS = {
    'michaelis_menten_irreversible': {
        'params': ['Vmax', 'Km'],
        'n_params': 2,
        'requires_inhibitor': False,
        'requires_two_substrates': False,
    },
    'michaelis_menten_reversible': {
        'params': ['Vmax_f', 'Km_s', 'Vmax_r', 'Km_p'],
        'n_params': 4,
        'requires_inhibitor': False,
        'requires_two_substrates': False,
    },
    'competitive_inhibition': {
        'params': ['Vmax', 'Km', 'Ki'],
        'n_params': 3,
        'requires_inhibitor': True,
        'requires_two_substrates': False,
    },
    'uncompetitive_inhibition': {
        'params': ['Vmax', 'Km', 'Ki'],
        'n_params': 3,
        'requires_inhibitor': True,
        'requires_two_substrates': False,
    },
    'mixed_inhibition': {
        'params': ['Vmax', 'Km', 'Ki', 'Ki_prime'],
        'n_params': 4,
        'requires_inhibitor': True,
        'requires_two_substrates': False,
    },
    'substrate_inhibition': {
        'params': ['Vmax', 'Km', 'Ki_S'],
        'n_params': 3,
        'requires_inhibitor': False,
        'requires_two_substrates': False,
    },
    'ordered_bi_bi': {
        'params': ['Vmax', 'Km_A', 'Km_B', 'Ki_A'],
        'n_params': 4,
        'requires_inhibitor': False,
        'requires_two_substrates': True,
    },
    'random_bi_bi': {
        'params': ['Vmax', 'Km_A', 'Km_B'],
        'n_params': 3,
        'requires_inhibitor': False,
        'requires_two_substrates': True,
    },
    'ping_pong': {
        'params': ['Vmax', 'Km_A', 'Km_B'],
        'n_params': 3,
        'requires_inhibitor': False,
        'requires_two_substrates': True,
    },
    'product_inhibition': {
        'params': ['Vmax', 'Km', 'Ki_P'],
        'n_params': 3,
        'requires_inhibitor': False,
        'requires_two_substrates': False,
    },
}

MECHANISMS = list(MECHANISM_SPECS.keys())


@dataclass
class FitResult:
    """Result of fitting a mechanism to data."""
    mechanism: str
    params: Optional[np.ndarray]
    param_names: List[str]
    ss_res: float  # Sum of squared residuals
    n_data: int    # Number of data points
    n_params: int  # Number of parameters
    aic: float     # Akaike Information Criterion
    bic: float     # Bayesian Information Criterion
    success: bool
    message: str = ""


class MechanismFitter:
    """
    Fits enzyme kinetic mechanisms to multi-condition data.

    For each mechanism:
    1. Define ODE system with parameters to fit
    2. Simulate mechanism with candidate parameters
    3. Minimize sum of squared residuals across all conditions
    4. Calculate AIC/BIC
    """

    def __init__(self, max_iter: int = 1000, n_restarts: int = 3):
        self.max_iter = max_iter
        self.n_restarts = n_restarts

        # Parameter bounds (log10 scale for better optimization)
        # Vmax: 1e-4 to 1e4
        # Km, Ki: 1e-4 to 1e2
        self.default_bounds = {
            'Vmax': (-4, 4),      # log10 scale
            'Vmax_f': (-4, 4),
            'Vmax_r': (-4, 4),
            'Km': (-4, 2),
            'Km_s': (-4, 2),
            'Km_p': (-4, 2),
            'Km_A': (-4, 2),
            'Km_B': (-4, 2),
            'Ki': (-4, 2),
            'Ki_prime': (-4, 2),
            'Ki_S': (-4, 2),
            'Ki_A': (-4, 2),
            'Ki_P': (-4, 2),
        }

    def fit_mechanism(self, sample: MultiConditionSample, mechanism: str) -> FitResult:
        """
        Fit a mechanism to multi-condition data.

        Parameters are fit GLOBALLY across all conditions (same enzyme).
        """
        spec = MECHANISM_SPECS[mechanism]
        param_names = spec['params']
        n_params = spec['n_params']

        # Check if this mechanism is compatible with the data
        has_inhibitor = any('I0' in traj['conditions'] and traj['conditions']['I0'] > 0
                          for traj in sample.trajectories)
        has_two_substrates = any('A0' in traj['conditions'] and 'B0' in traj['conditions']
                                for traj in sample.trajectories)

        if spec['requires_inhibitor'] and not has_inhibitor:
            # Can't fit inhibition mechanism to data without inhibitor
            return FitResult(
                mechanism=mechanism,
                params=None,
                param_names=param_names,
                ss_res=np.inf,
                n_data=0,
                n_params=n_params,
                aic=np.inf,
                bic=np.inf,
                success=False,
                message="Data lacks inhibitor variation"
            )

        if spec['requires_two_substrates'] and not has_two_substrates:
            return FitResult(
                mechanism=mechanism,
                params=None,
                param_names=param_names,
                ss_res=np.inf,
                n_data=0,
                n_params=n_params,
                aic=np.inf,
                bic=np.inf,
                success=False,
                message="Data lacks two-substrate variation"
            )

        if not spec['requires_two_substrates'] and has_two_substrates:
            # Single substrate mechanism can't fit two-substrate data
            return FitResult(
                mechanism=mechanism,
                params=None,
                param_names=param_names,
                ss_res=np.inf,
                n_data=0,
                n_params=n_params,
                aic=np.inf,
                bic=np.inf,
                success=False,
                message="Data has two substrates, mechanism expects one"
            )

        # Get bounds for optimization
        bounds = [self.default_bounds[p] for p in param_names]

        # Collect all data points
        all_t = []
        all_y_obs = []
        all_conditions = []

        for traj in sample.trajectories:
            t = traj['t']
            # Use substrate concentration as target (primary observable)
            if 'S' in traj['concentrations']:
                y = traj['concentrations']['S']
            elif 'A' in traj['concentrations']:
                y = traj['concentrations']['A']
            else:
                continue

            all_t.append(t)
            all_y_obs.append(y)
            all_conditions.append(traj['conditions'])

        if len(all_t) == 0:
            return FitResult(
                mechanism=mechanism,
                params=None,
                param_names=param_names,
                ss_res=np.inf,
                n_data=0,
                n_params=n_params,
                aic=np.inf,
                bic=np.inf,
                success=False,
                message="No valid trajectories"
            )

        n_data = sum(len(y) for y in all_y_obs)

        # Define objective function
        def objective(params_log):
            params = 10 ** np.array(params_log)  # Convert from log scale
            total_ss = 0.0

            for t, y_obs, conditions in zip(all_t, all_y_obs, all_conditions):
                try:
                    y_pred = self._simulate_mechanism(mechanism, params, param_names, t, conditions)
                    if y_pred is None:
                        return 1e10
                    ss = np.sum((y_obs - y_pred) ** 2)
                    total_ss += ss
                except Exception:
                    return 1e10

            return total_ss

        # Multi-start optimization
        best_result = None
        best_ss = np.inf

        for restart in range(self.n_restarts):
            # Random initial guess within bounds
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': self.max_iter, 'disp': False}
                    )

                if result.fun < best_ss:
                    best_ss = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None or best_ss >= 1e9:
            return FitResult(
                mechanism=mechanism,
                params=None,
                param_names=param_names,
                ss_res=np.inf,
                n_data=n_data,
                n_params=n_params,
                aic=np.inf,
                bic=np.inf,
                success=False,
                message="Optimization failed"
            )

        # Calculate AIC and BIC
        ss_res = best_ss

        # AIC = n * ln(SS/n) + 2k
        # BIC = n * ln(SS/n) + k * ln(n)
        if ss_res > 0 and n_data > n_params:
            aic = n_data * np.log(ss_res / n_data) + 2 * n_params
            bic = n_data * np.log(ss_res / n_data) + n_params * np.log(n_data)
        else:
            aic = np.inf
            bic = np.inf

        return FitResult(
            mechanism=mechanism,
            params=10 ** best_result.x,  # Convert back from log scale
            param_names=param_names,
            ss_res=ss_res,
            n_data=n_data,
            n_params=n_params,
            aic=aic,
            bic=bic,
            success=True
        )

    def _simulate_mechanism(self, mechanism: str, params: np.ndarray,
                           param_names: List[str], t: np.ndarray,
                           conditions: Dict) -> Optional[np.ndarray]:
        """Simulate a mechanism with given parameters."""

        # Convert params array to dict
        p = {name: val for name, val in zip(param_names, params)}

        # Get initial conditions
        E0 = conditions.get('E0', 1e-3)

        try:
            if mechanism == 'michaelis_menten_irreversible':
                S0 = conditions['S0']
                def ode(y, t):
                    S = max(y[0], 0)
                    v = p['Vmax'] * S / (p['Km'] + S)
                    return [-v]
                sol = odeint(ode, [S0], t)
                return sol[:, 0]

            elif mechanism == 'michaelis_menten_reversible':
                S0 = conditions['S0']
                P0 = conditions.get('P0', 0.0)
                def ode(y, t):
                    S, P = max(y[0], 0), max(y[1], 0)
                    v_fwd = p['Vmax_f'] * S / (p['Km_s'] + S)
                    v_rev = p['Vmax_r'] * P / (p['Km_p'] + P)
                    return [-v_fwd + v_rev, v_fwd - v_rev]
                sol = odeint(ode, [S0, P0], t)
                return sol[:, 0]

            elif mechanism == 'competitive_inhibition':
                S0 = conditions['S0']
                I0 = conditions.get('I0', 0.0)
                def ode(y, t):
                    S = max(y[0], 0)
                    Km_app = p['Km'] * (1 + I0 / p['Ki'])
                    v = p['Vmax'] * S / (Km_app + S)
                    return [-v]
                sol = odeint(ode, [S0], t)
                return sol[:, 0]

            elif mechanism == 'uncompetitive_inhibition':
                S0 = conditions['S0']
                I0 = conditions.get('I0', 0.0)
                def ode(y, t):
                    S = max(y[0], 0)
                    alpha_prime = 1 + I0 / p['Ki']
                    Vmax_app = p['Vmax'] / alpha_prime
                    Km_app = p['Km'] / alpha_prime
                    v = Vmax_app * S / (Km_app + S)
                    return [-v]
                sol = odeint(ode, [S0], t)
                return sol[:, 0]

            elif mechanism == 'mixed_inhibition':
                S0 = conditions['S0']
                I0 = conditions.get('I0', 0.0)
                def ode(y, t):
                    S = max(y[0], 0)
                    alpha = 1 + I0 / p['Ki']
                    alpha_prime = 1 + I0 / p['Ki_prime']
                    Vmax_app = p['Vmax'] / alpha_prime
                    Km_app = p['Km'] * alpha / alpha_prime
                    v = Vmax_app * S / (Km_app + S)
                    return [-v]
                sol = odeint(ode, [S0], t)
                return sol[:, 0]

            elif mechanism == 'substrate_inhibition':
                S0 = conditions['S0']
                def ode(y, t):
                    S = max(y[0], 0)
                    v = p['Vmax'] * S / (p['Km'] + S + S**2 / p['Ki_S'])
                    return [-v]
                sol = odeint(ode, [S0], t)
                return sol[:, 0]

            elif mechanism == 'ordered_bi_bi':
                A0, B0 = conditions['A0'], conditions['B0']
                def ode(y, t):
                    A, B = max(y[0], 0), max(y[1], 0)
                    denom = p['Ki_A'] * p['Km_B'] + p['Km_B'] * A + p['Km_A'] * B + A * B
                    v = p['Vmax'] * A * B / (denom + 1e-10)
                    return [-v, -v]
                sol = odeint(ode, [A0, B0], t)
                return sol[:, 0]

            elif mechanism == 'random_bi_bi':
                A0, B0 = conditions['A0'], conditions['B0']
                def ode(y, t):
                    A, B = max(y[0], 0), max(y[1], 0)
                    denom = p['Km_A'] * p['Km_B'] + p['Km_B'] * A + p['Km_A'] * B + A * B
                    v = p['Vmax'] * A * B / (denom + 1e-10)
                    return [-v, -v]
                sol = odeint(ode, [A0, B0], t)
                return sol[:, 0]

            elif mechanism == 'ping_pong':
                A0, B0 = conditions['A0'], conditions['B0']
                def ode(y, t):
                    A, B = max(y[0], 0), max(y[1], 0)
                    denom = p['Km_A'] * B + p['Km_B'] * A + A * B
                    v = p['Vmax'] * A * B / (denom + 1e-10)
                    return [-v, -v]
                sol = odeint(ode, [A0, B0], t)
                return sol[:, 0]

            elif mechanism == 'product_inhibition':
                S0 = conditions['S0']
                P0 = conditions.get('P0', 0.0)
                def ode(y, t):
                    S, P = max(y[0], 0), max(y[1], 0)
                    Km_app = p['Km'] * (1 + P / p['Ki_P'])
                    v = p['Vmax'] * S / (Km_app + S)
                    return [-v, v]
                sol = odeint(ode, [S0, P0], t)
                return sol[:, 0]

            else:
                return None

        except Exception:
            return None


class ClassicalModelSelector:
    """
    Classical model selection using AIC/BIC.

    This is what biochemists actually do (minus the ML).
    """

    def __init__(self, criterion: str = 'aic'):
        """
        Args:
            criterion: 'aic' or 'bic' for model selection
        """
        self.criterion = criterion
        self.fitter = MechanismFitter()

    def predict(self, sample: MultiConditionSample) -> Tuple[str, Dict[str, FitResult]]:
        """
        Predict mechanism using classical model selection.

        Returns:
            (predicted_mechanism, {mechanism: FitResult})
        """
        results = {}

        for mechanism in MECHANISMS:
            result = self.fitter.fit_mechanism(sample, mechanism)
            results[mechanism] = result

        # Select best mechanism by AIC or BIC
        if self.criterion == 'aic':
            best_mech = min(results.keys(), key=lambda m: results[m].aic)
        else:
            best_mech = min(results.keys(), key=lambda m: results[m].bic)

        return best_mech, results


def evaluate_classical_baseline(samples: List[MultiConditionSample],
                               criterion: str = 'aic',
                               verbose: bool = True) -> Dict:
    """
    Evaluate classical model selection on a set of samples.

    Returns accuracy and per-mechanism statistics.
    """
    selector = ClassicalModelSelector(criterion=criterion)

    correct = 0
    total = 0
    per_mechanism = {m: {'correct': 0, 'total': 0} for m in MECHANISMS}
    confusion = {m1: {m2: 0 for m2 in MECHANISMS} for m1 in MECHANISMS}
    predictions = []
    labels = []

    for i, sample in enumerate(samples):
        if verbose and i % 50 == 0:
            print(f"  Classical fitting: {i}/{len(samples)}")

        true_mech = sample.mechanism
        pred_mech, _ = selector.predict(sample)

        # Track predictions and labels
        true_idx = MECHANISMS.index(true_mech)
        pred_idx = MECHANISMS.index(pred_mech)
        predictions.append(pred_idx)
        labels.append(true_idx)

        per_mechanism[true_mech]['total'] += 1
        confusion[true_mech][pred_mech] += 1

        if pred_mech == true_mech:
            correct += 1
            per_mechanism[true_mech]['correct'] += 1

        total += 1

    accuracy = correct / total if total > 0 else 0

    # Calculate per-mechanism accuracy
    per_mech_acc = {}
    for m in MECHANISMS:
        if per_mechanism[m]['total'] > 0:
            per_mech_acc[m] = per_mechanism[m]['correct'] / per_mechanism[m]['total']
        else:
            per_mech_acc[m] = 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_mechanism': per_mech_acc,
        'confusion': confusion,
        'criterion': criterion,
        'predictions': predictions,
        'labels': labels,
    }


def main():
    """Demo evaluation on a small test set."""
    import argparse

    parser = argparse.ArgumentParser(description='Classical AIC/BIC baseline')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Samples per mechanism')
    parser.add_argument('--criterion', type=str, default='aic',
                       choices=['aic', 'bic'], help='Model selection criterion')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed')
    args = parser.parse_args()

    print("="*70)
    print("Classical Model Selection Baseline (AIC/BIC)")
    print("="*70)

    # Generate test samples
    print(f"\nGenerating {args.n_samples} samples per mechanism...")
    from tactic_kinetics.training.multi_condition_generator import (
        MultiConditionGenerator, MultiConditionConfig
    )

    config = MultiConditionConfig(n_conditions_per_sample=20)
    generator = MultiConditionGenerator(config, seed=args.seed)
    samples = generator.generate_batch(args.n_samples, n_workers=1)

    print(f"Generated {len(samples)} samples")

    # Evaluate
    print(f"\nEvaluating with {args.criterion.upper()}...")
    results = evaluate_classical_baseline(samples, criterion=args.criterion)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall accuracy: {results['accuracy']*100:.1f}%")
    print(f"({results['correct']}/{results['total']} correct)")

    print("\nPer-mechanism accuracy:")
    for mech, acc in sorted(results['per_mechanism'].items(), key=lambda x: -x[1]):
        print(f"  {mech:35s}: {acc*100:5.1f}%")

    return results


if __name__ == "__main__":
    main()
