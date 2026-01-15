"""
Predefined enzyme mechanism templates.

This module defines the 6 mechanism families mentioned in the TACTIC-Kinetics
project, plus additional common mechanisms.
"""

from typing import Dict, List, Optional, Type
from .base import MechanismTemplate, State, Transition, StateType, TransitionType


# Registry of all mechanisms
MECHANISM_REGISTRY: Dict[str, MechanismTemplate] = {}


def register_mechanism(mechanism: MechanismTemplate) -> MechanismTemplate:
    """Register a mechanism in the global registry."""
    MECHANISM_REGISTRY[mechanism.name] = mechanism
    return mechanism


def get_mechanism_by_name(name: str) -> MechanismTemplate:
    """Get a mechanism by name from the registry."""
    if name not in MECHANISM_REGISTRY:
        raise KeyError(f"Unknown mechanism: {name}. Available: {list(MECHANISM_REGISTRY.keys())}")
    return MECHANISM_REGISTRY[name]


def get_all_mechanisms() -> Dict[str, MechanismTemplate]:
    """Get all registered mechanisms."""
    return MECHANISM_REGISTRY.copy()


# =============================================================================
# 1. Michaelis-Menten Mechanisms
# =============================================================================

MichaelisMentenIrreversible = register_mechanism(MechanismTemplate(
    name="michaelis_menten_irreversible",
    description="Irreversible Michaelis-Menten: E + S <-> ES -> E + P",
    states=[
        State(
            name="E_S",
            state_type=StateType.FREE_ENZYME,
            species=["E", "S"],
            is_reference=True,
        ),
        State(
            name="ES",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["ES"],
            energy_param_name="G_ES",
        ),
        State(
            name="E_P",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P"],
            energy_param_name="G_rxn",  # ΔG°_rxn
        ),
    ],
    transitions=[
        Transition(
            name="binding",
            from_state="E_S",
            to_state="ES",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="catalysis",
            from_state="ES",
            to_state="E_P",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=False,
            stoichiometry={"P": 1},
        ),
    ],
    substrate_names=["S"],
    product_names=["P"],
))


MichaelisMentenReversible = register_mechanism(MechanismTemplate(
    name="michaelis_menten_reversible",
    description="Reversible Michaelis-Menten: E + S <-> ES <-> EP <-> E + P",
    states=[
        State(
            name="E_S",
            state_type=StateType.FREE_ENZYME,
            species=["E", "S"],
            is_reference=True,
        ),
        State(
            name="ES",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["ES"],
            energy_param_name="G_ES",
        ),
        State(
            name="EP",
            state_type=StateType.ENZYME_PRODUCT,
            species=["EP"],
            energy_param_name="G_EP",
        ),
        State(
            name="E_P",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="substrate_binding",
            from_state="E_S",
            to_state="ES",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="catalysis",
            from_state="ES",
            to_state="EP",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=True,
        ),
        Transition(
            name="product_release",
            from_state="EP",
            to_state="E_P",
            transition_type=TransitionType.RELEASE,
            barrier_param_name="G_barrier_release_P",
            is_reversible=True,
            stoichiometry={"P": 1},
        ),
    ],
    substrate_names=["S"],
    product_names=["P"],
))


# =============================================================================
# 2. Inhibition Mechanisms
# =============================================================================

CompetitiveInhibition = register_mechanism(MechanismTemplate(
    name="competitive_inhibition",
    description="Competitive inhibition: I competes with S for binding to E",
    states=[
        State(
            name="E_S",
            state_type=StateType.FREE_ENZYME,
            species=["E", "S"],
            is_reference=True,
        ),
        State(
            name="ES",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["ES"],
            energy_param_name="G_ES",
        ),
        State(
            name="EI",
            state_type=StateType.ENZYME_INHIBITOR,
            species=["EI"],
            energy_param_name="G_EI",
        ),
        State(
            name="E_P",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="substrate_binding",
            from_state="E_S",
            to_state="ES",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="inhibitor_binding",
            from_state="E_S",
            to_state="EI",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_I",
            is_reversible=True,
            stoichiometry={"S": -1, "I": -1},  # E + I -> EI (S released implicitly)
        ),
        Transition(
            name="catalysis",
            from_state="ES",
            to_state="E_P",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=False,
            stoichiometry={"P": 1},
        ),
    ],
    substrate_names=["S"],
    product_names=["P"],
    inhibitor_names=["I"],
))


UncompetitiveInhibition = register_mechanism(MechanismTemplate(
    name="uncompetitive_inhibition",
    description="Uncompetitive inhibition: I binds only to ES complex",
    states=[
        State(
            name="E_S",
            state_type=StateType.FREE_ENZYME,
            species=["E", "S"],
            is_reference=True,
        ),
        State(
            name="ES",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["ES"],
            energy_param_name="G_ES",
        ),
        State(
            name="ESI",
            state_type=StateType.TERNARY_SUBSTRATE,
            species=["ESI"],
            energy_param_name="G_ESI",
        ),
        State(
            name="E_P",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="substrate_binding",
            from_state="E_S",
            to_state="ES",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="inhibitor_binding",
            from_state="ES",
            to_state="ESI",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_I",
            is_reversible=True,
            stoichiometry={"I": -1},
        ),
        Transition(
            name="catalysis",
            from_state="ES",
            to_state="E_P",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=False,
            stoichiometry={"P": 1},
        ),
    ],
    substrate_names=["S"],
    product_names=["P"],
    inhibitor_names=["I"],
))


MixedInhibition = register_mechanism(MechanismTemplate(
    name="mixed_inhibition",
    description="Mixed (noncompetitive) inhibition: I can bind to E or ES",
    states=[
        State(
            name="E_S",
            state_type=StateType.FREE_ENZYME,
            species=["E", "S"],
            is_reference=True,
        ),
        State(
            name="ES",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["ES"],
            energy_param_name="G_ES",
        ),
        State(
            name="EI",
            state_type=StateType.ENZYME_INHIBITOR,
            species=["EI"],
            energy_param_name="G_EI",
        ),
        State(
            name="ESI",
            state_type=StateType.TERNARY_SUBSTRATE,
            species=["ESI"],
            energy_param_name="G_ESI",
        ),
        State(
            name="E_P",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="substrate_binding",
            from_state="E_S",
            to_state="ES",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="inhibitor_binding_E",
            from_state="E_S",
            to_state="EI",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_I_E",
            is_reversible=True,
            stoichiometry={"I": -1},
        ),
        Transition(
            name="inhibitor_binding_ES",
            from_state="ES",
            to_state="ESI",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_I_ES",
            is_reversible=True,
            stoichiometry={"I": -1},
        ),
        Transition(
            name="substrate_binding_EI",
            from_state="EI",
            to_state="ESI",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S_EI",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="catalysis",
            from_state="ES",
            to_state="E_P",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=False,
            stoichiometry={"P": 1},
        ),
    ],
    substrate_names=["S"],
    product_names=["P"],
    inhibitor_names=["I"],
))


SubstrateInhibition = register_mechanism(MechanismTemplate(
    name="substrate_inhibition",
    description="Substrate inhibition: excess S binds to ES forming inactive ESS",
    states=[
        State(
            name="E_S",
            state_type=StateType.FREE_ENZYME,
            species=["E", "S"],
            is_reference=True,
        ),
        State(
            name="ES",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["ES"],
            energy_param_name="G_ES",
        ),
        State(
            name="ESS",
            state_type=StateType.TERNARY_SUBSTRATE,
            species=["ESS"],
            energy_param_name="G_ESS",
        ),
        State(
            name="E_P",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="substrate_binding",
            from_state="E_S",
            to_state="ES",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="substrate_inhibition",
            from_state="ES",
            to_state="ESS",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S2",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="catalysis",
            from_state="ES",
            to_state="E_P",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=False,
            stoichiometry={"P": 1},
        ),
    ],
    substrate_names=["S"],
    product_names=["P"],
))


# =============================================================================
# 3. Bi-Substrate Mechanisms
# =============================================================================

OrderedBiBi = register_mechanism(MechanismTemplate(
    name="ordered_bi_bi",
    description="Ordered Bi-Bi: A binds first, Q releases last. E + A + B <-> EA + B <-> EAB <-> EPQ <-> EQ + P <-> E + P + Q",
    states=[
        State(
            name="E_AB",
            state_type=StateType.FREE_ENZYME,
            species=["E", "A", "B"],
            is_reference=True,
        ),
        State(
            name="EA_B",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["EA", "B"],
            energy_param_name="G_EA",
        ),
        State(
            name="EAB",
            state_type=StateType.TERNARY_SUBSTRATE,
            species=["EAB"],
            energy_param_name="G_EAB",
        ),
        State(
            name="EPQ",
            state_type=StateType.TERNARY_PRODUCT,
            species=["EPQ"],
            energy_param_name="G_EPQ",
        ),
        State(
            name="EQ_P",
            state_type=StateType.ENZYME_PRODUCT,
            species=["EQ", "P"],
            energy_param_name="G_EQ",
        ),
        State(
            name="E_PQ",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P", "Q"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="A_binding",
            from_state="E_AB",
            to_state="EA_B",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_A",
            is_reversible=True,
            stoichiometry={"A": -1},
        ),
        Transition(
            name="B_binding",
            from_state="EA_B",
            to_state="EAB",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_B",
            is_reversible=True,
            stoichiometry={"B": -1},
        ),
        Transition(
            name="catalysis",
            from_state="EAB",
            to_state="EPQ",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=True,
        ),
        Transition(
            name="P_release",
            from_state="EPQ",
            to_state="EQ_P",
            transition_type=TransitionType.RELEASE,
            barrier_param_name="G_barrier_release_P",
            is_reversible=True,
            stoichiometry={"P": 1},
        ),
        Transition(
            name="Q_release",
            from_state="EQ_P",
            to_state="E_PQ",
            transition_type=TransitionType.RELEASE,
            barrier_param_name="G_barrier_release_Q",
            is_reversible=True,
            stoichiometry={"Q": 1},
        ),
    ],
    substrate_names=["A", "B"],
    product_names=["P", "Q"],
))


RandomBiBi = register_mechanism(MechanismTemplate(
    name="random_bi_bi",
    description="Random Bi-Bi: Either A or B can bind first",
    states=[
        State(
            name="E_AB",
            state_type=StateType.FREE_ENZYME,
            species=["E", "A", "B"],
            is_reference=True,
        ),
        State(
            name="EA_B",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["EA", "B"],
            energy_param_name="G_EA",
        ),
        State(
            name="EB_A",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["EB", "A"],
            energy_param_name="G_EB",
        ),
        State(
            name="EAB",
            state_type=StateType.TERNARY_SUBSTRATE,
            species=["EAB"],
            energy_param_name="G_EAB",
        ),
        State(
            name="EPQ",
            state_type=StateType.TERNARY_PRODUCT,
            species=["EPQ"],
            energy_param_name="G_EPQ",
        ),
        State(
            name="E_PQ",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P", "Q"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="A_binding_first",
            from_state="E_AB",
            to_state="EA_B",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_A",
            is_reversible=True,
            stoichiometry={"A": -1},
        ),
        Transition(
            name="B_binding_first",
            from_state="E_AB",
            to_state="EB_A",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_B",
            is_reversible=True,
            stoichiometry={"B": -1},
        ),
        Transition(
            name="B_binding_second",
            from_state="EA_B",
            to_state="EAB",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_B_EA",
            is_reversible=True,
            stoichiometry={"B": -1},
        ),
        Transition(
            name="A_binding_second",
            from_state="EB_A",
            to_state="EAB",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_A_EB",
            is_reversible=True,
            stoichiometry={"A": -1},
        ),
        Transition(
            name="catalysis",
            from_state="EAB",
            to_state="EPQ",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=True,
        ),
        Transition(
            name="product_release",
            from_state="EPQ",
            to_state="E_PQ",
            transition_type=TransitionType.RELEASE,
            barrier_param_name="G_barrier_release",
            is_reversible=True,
            stoichiometry={"P": 1, "Q": 1},
        ),
    ],
    substrate_names=["A", "B"],
    product_names=["P", "Q"],
))


PingPong = register_mechanism(MechanismTemplate(
    name="ping_pong",
    description="Ping-Pong (double displacement): E + A <-> EA <-> FP <-> F + P; F + B <-> FB <-> EQ <-> E + Q",
    states=[
        State(
            name="E_A",
            state_type=StateType.FREE_ENZYME,
            species=["E", "A"],
            is_reference=True,
        ),
        State(
            name="EA",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["EA"],
            energy_param_name="G_EA",
        ),
        State(
            name="FP",
            state_type=StateType.ENZYME_PRODUCT,
            species=["FP"],
            energy_param_name="G_FP",
        ),
        State(
            name="F_P",
            state_type=StateType.FREE_ENZYME,
            species=["F", "P"],
            energy_param_name="G_F",
        ),
        State(
            name="F_B",
            state_type=StateType.FREE_ENZYME,
            species=["F", "B"],
            energy_param_name="G_F_B",
        ),
        State(
            name="FB",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["FB"],
            energy_param_name="G_FB",
        ),
        State(
            name="EQ",
            state_type=StateType.ENZYME_PRODUCT,
            species=["EQ"],
            energy_param_name="G_EQ",
        ),
        State(
            name="E_Q",
            state_type=StateType.FREE_ENZYME,
            species=["E", "Q"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="A_binding",
            from_state="E_A",
            to_state="EA",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_A",
            is_reversible=True,
            stoichiometry={"A": -1},
        ),
        Transition(
            name="first_catalysis",
            from_state="EA",
            to_state="FP",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat1",
            is_reversible=True,
        ),
        Transition(
            name="P_release",
            from_state="FP",
            to_state="F_P",
            transition_type=TransitionType.RELEASE,
            barrier_param_name="G_barrier_release_P",
            is_reversible=True,
            stoichiometry={"P": 1},
        ),
        Transition(
            name="B_binding",
            from_state="F_B",
            to_state="FB",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_B",
            is_reversible=True,
            stoichiometry={"B": -1},
        ),
        Transition(
            name="second_catalysis",
            from_state="FB",
            to_state="EQ",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat2",
            is_reversible=True,
        ),
        Transition(
            name="Q_release",
            from_state="EQ",
            to_state="E_Q",
            transition_type=TransitionType.RELEASE,
            barrier_param_name="G_barrier_release_Q",
            is_reversible=True,
            stoichiometry={"Q": 1},
        ),
    ],
    substrate_names=["A", "B"],
    product_names=["P", "Q"],
))


# =============================================================================
# 4. Product Inhibition
# =============================================================================

ProductInhibition = register_mechanism(MechanismTemplate(
    name="product_inhibition",
    description="Product inhibition: P can bind back to E competitively",
    states=[
        State(
            name="E_S",
            state_type=StateType.FREE_ENZYME,
            species=["E", "S"],
            is_reference=True,
        ),
        State(
            name="ES",
            state_type=StateType.ENZYME_SUBSTRATE,
            species=["ES"],
            energy_param_name="G_ES",
        ),
        State(
            name="EP",
            state_type=StateType.ENZYME_PRODUCT,
            species=["EP"],
            energy_param_name="G_EP",
        ),
        State(
            name="E_P",
            state_type=StateType.FREE_ENZYME,
            species=["E", "P"],
            energy_param_name="G_rxn",
        ),
    ],
    transitions=[
        Transition(
            name="substrate_binding",
            from_state="E_S",
            to_state="ES",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_S",
            is_reversible=True,
            stoichiometry={"S": -1},
        ),
        Transition(
            name="catalysis",
            from_state="ES",
            to_state="EP",
            transition_type=TransitionType.CATALYSIS,
            barrier_param_name="G_barrier_cat",
            is_reversible=True,
        ),
        Transition(
            name="product_release",
            from_state="EP",
            to_state="E_P",
            transition_type=TransitionType.RELEASE,
            barrier_param_name="G_barrier_release",
            is_reversible=True,
            stoichiometry={"P": 1},
        ),
        Transition(
            name="product_rebinding",
            from_state="E_P",
            to_state="EP",
            transition_type=TransitionType.BINDING,
            barrier_param_name="G_barrier_bind_P",
            is_reversible=True,
            stoichiometry={"P": -1},
        ),
    ],
    substrate_names=["S"],
    product_names=["P"],
))


# Register product inhibition
MECHANISM_REGISTRY["product_inhibition"] = ProductInhibition


def list_mechanisms() -> None:
    """Print a summary of all available mechanisms."""
    print("Available Enzyme Mechanisms:")
    print("-" * 60)
    for name, mech in MECHANISM_REGISTRY.items():
        print(f"  {name}:")
        print(f"    States: {mech.n_states}, Transitions: {mech.n_transitions}")
        print(f"    Parameters: {mech.n_total_params}")
        print(f"    {mech.description}")
        print()
