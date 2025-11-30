"""State representations for games."""

from dataclasses import dataclass

CellType = None | int


@dataclass(frozen=True, slots=True)
class State:
    """State representation for 2x2 SLOB game."""

    board: tuple[CellType, ...]
    x_chips: int
    o_chips: int
    advantage: int


def canonicalize_state(state: State, canonical_map: dict[tuple, State]) -> State:
    """Return the canonicalized version of the state.

    Args:
        state: The game state.
        canonical_map: Map of states to their canonical version.

    Returns:
        Canonical state.

    """
    if state.board not in canonical_map:
        return state
    return canonical_map[state.board]
