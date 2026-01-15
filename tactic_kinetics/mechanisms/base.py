"""
Base classes for enzyme mechanism definitions.

A mechanism is defined as a graph of states connected by transitions.
Each state has an associated Gibbs energy, and each transition has
an activation Gibbs energy (barrier height).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto
import torch
import numpy as np


class StateType(Enum):
    """Types of states in an enzyme mechanism."""
    FREE_ENZYME = auto()          # E (free enzyme)
    ENZYME_SUBSTRATE = auto()     # ES, EA, EB, etc.
    ENZYME_PRODUCT = auto()       # EP, EQ, etc.
    ENZYME_INHIBITOR = auto()     # EI
    TERNARY_SUBSTRATE = auto()    # EAB, ESI, etc.
    TERNARY_PRODUCT = auto()      # EPQ
    FREE_SPECIES = auto()         # S, P, I (free in solution)


class TransitionType(Enum):
    """Types of transitions in an enzyme mechanism."""
    BINDING = auto()              # Substrate/inhibitor binding
    RELEASE = auto()              # Product/substrate release
    CATALYSIS = auto()            # Chemical transformation
    CONFORMATIONAL = auto()       # Conformational change


@dataclass
class State:
    """
    Represents a state in the enzyme mechanism.

    Attributes:
        name: Unique identifier for the state (e.g., "ES", "EI")
        state_type: Type of state (free enzyme, complex, etc.)
        species: List of species present in this state
        energy_param_name: Name of the energy parameter for this state
        is_reference: Whether this is the reference state (energy = 0)
    """
    name: str
    state_type: StateType
    species: List[str] = field(default_factory=list)
    energy_param_name: Optional[str] = None
    is_reference: bool = False

    def __post_init__(self):
        if self.energy_param_name is None:
            self.energy_param_name = f"G_{self.name}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name
        return False


@dataclass
class Transition:
    """
    Represents a transition between two states.

    Attributes:
        name: Unique identifier for the transition
        from_state: Starting state name
        to_state: Ending state name
        transition_type: Type of transition
        barrier_param_name: Name of the activation energy parameter
        is_reversible: Whether the transition is reversible
        stoichiometry: Dict mapping species to stoichiometric coefficients
                      (negative for consumed, positive for produced)
    """
    name: str
    from_state: str
    to_state: str
    transition_type: TransitionType
    barrier_param_name: Optional[str] = None
    is_reversible: bool = True
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.barrier_param_name is None:
            self.barrier_param_name = f"G_barrier_{self.name}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Transition):
            return self.name == other.name
        return False


@dataclass
class MechanismTemplate:
    """
    Template for an enzyme mechanism.

    Defines the graph structure (states and transitions) and provides
    methods for computing rate constants from energy parameters.

    Attributes:
        name: Name of the mechanism
        states: List of State objects
        transitions: List of Transition objects
        substrate_names: Names of substrate species
        product_names: Names of product species
        inhibitor_names: Names of inhibitor species (if any)
        n_energy_params: Number of energy parameters
        n_barrier_params: Number of activation energy parameters
    """
    name: str
    states: List[State]
    transitions: List[Transition]
    substrate_names: List[str] = field(default_factory=list)
    product_names: List[str] = field(default_factory=list)
    inhibitor_names: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        self._state_dict = {s.name: s for s in self.states}
        self._transition_dict = {t.name: t for t in self.transitions}
        self._validate()

    def _validate(self):
        """Validate the mechanism definition."""
        # Check that all transitions reference valid states
        state_names = set(s.name for s in self.states)
        for t in self.transitions:
            if t.from_state not in state_names:
                raise ValueError(f"Transition {t.name} references unknown state {t.from_state}")
            if t.to_state not in state_names:
                raise ValueError(f"Transition {t.name} references unknown state {t.to_state}")

        # Check that exactly one state is the reference
        reference_states = [s for s in self.states if s.is_reference]
        if len(reference_states) != 1:
            raise ValueError(f"Mechanism must have exactly one reference state, found {len(reference_states)}")

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    @property
    def n_energy_params(self) -> int:
        """Number of state energy parameters (excluding reference)."""
        return len([s for s in self.states if not s.is_reference])

    @property
    def n_barrier_params(self) -> int:
        """Number of activation energy parameters."""
        return len(self.transitions)

    @property
    def n_total_params(self) -> int:
        """Total number of energy parameters."""
        return self.n_energy_params + self.n_barrier_params

    @property
    def energy_param_names(self) -> List[str]:
        """Names of all state energy parameters."""
        return [s.energy_param_name for s in self.states if not s.is_reference]

    @property
    def barrier_param_names(self) -> List[str]:
        """Names of all activation energy parameters."""
        return [t.barrier_param_name for t in self.transitions]

    @property
    def all_param_names(self) -> List[str]:
        """Names of all energy parameters."""
        return self.energy_param_names + self.barrier_param_names

    @property
    def reference_state(self) -> State:
        """Get the reference state."""
        for s in self.states:
            if s.is_reference:
                return s
        raise ValueError("No reference state found")

    @property
    def species_names(self) -> List[str]:
        """All species involved in the mechanism."""
        species = set()
        for s in self.states:
            species.update(s.species)
        for t in self.transitions:
            species.update(t.stoichiometry.keys())
        return sorted(list(species))

    def get_state(self, name: str) -> State:
        """Get state by name."""
        return self._state_dict[name]

    def get_transition(self, name: str) -> Transition:
        """Get transition by name."""
        return self._transition_dict[name]

    def get_outgoing_transitions(self, state_name: str) -> List[Transition]:
        """Get all transitions leaving a state."""
        return [t for t in self.transitions if t.from_state == state_name]

    def get_incoming_transitions(self, state_name: str) -> List[Transition]:
        """Get all transitions entering a state."""
        return [t for t in self.transitions if t.to_state == state_name]

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the mechanism graph.

        Returns:
            A (n_states x n_states) matrix where entry [i,j] is 1 if
            there is a transition from state i to state j.
        """
        n = self.n_states
        adj = np.zeros((n, n))
        state_idx = {s.name: i for i, s in enumerate(self.states)}

        for t in self.transitions:
            i = state_idx[t.from_state]
            j = state_idx[t.to_state]
            adj[i, j] = 1
            if t.is_reversible:
                adj[j, i] = 1

        return adj

    def get_stoichiometry_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Get the stoichiometry matrix for the mechanism.

        Returns:
            Tuple of (matrix, state_names, species_names) where matrix[i,j]
            gives the stoichiometric coefficient of species j in state i.
        """
        species = self.species_names
        n_species = len(species)
        n_states = self.n_states

        S = np.zeros((n_states, n_species))
        species_idx = {s: i for i, s in enumerate(species)}
        state_names = [s.name for s in self.states]

        for i, state in enumerate(self.states):
            for sp in state.species:
                if sp in species_idx:
                    S[i, species_idx[sp]] = 1

        return S, state_names, species

    def to_dict(self) -> dict:
        """Convert mechanism to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "states": [
                {
                    "name": s.name,
                    "state_type": s.state_type.name,
                    "species": s.species,
                    "energy_param_name": s.energy_param_name,
                    "is_reference": s.is_reference,
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "name": t.name,
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "transition_type": t.transition_type.name,
                    "barrier_param_name": t.barrier_param_name,
                    "is_reversible": t.is_reversible,
                    "stoichiometry": t.stoichiometry,
                }
                for t in self.transitions
            ],
            "substrate_names": self.substrate_names,
            "product_names": self.product_names,
            "inhibitor_names": self.inhibitor_names,
        }

    def __repr__(self) -> str:
        return (
            f"MechanismTemplate(name='{self.name}', "
            f"n_states={self.n_states}, n_transitions={self.n_transitions}, "
            f"n_params={self.n_total_params})"
        )

    def describe(self) -> str:
        """Get a detailed description of the mechanism."""
        lines = [
            f"Mechanism: {self.name}",
            f"Description: {self.description}",
            "",
            "States:",
        ]
        for s in self.states:
            ref_marker = " (reference)" if s.is_reference else ""
            lines.append(f"  - {s.name}: {s.state_type.name}{ref_marker}")

        lines.append("")
        lines.append("Transitions:")
        for t in self.transitions:
            rev_marker = " (reversible)" if t.is_reversible else " (irreversible)"
            lines.append(f"  - {t.name}: {t.from_state} -> {t.to_state}{rev_marker}")

        lines.append("")
        lines.append(f"Energy parameters: {self.n_energy_params}")
        lines.append(f"Barrier parameters: {self.n_barrier_params}")
        lines.append(f"Total parameters: {self.n_total_params}")

        return "\n".join(lines)
