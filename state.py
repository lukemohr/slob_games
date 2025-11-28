"""State representations for games."""

from dataclasses import dataclass

CellType = None | int


@dataclass(frozen=True)
class State:
    """State representation for 2x2 SLOB game."""

    board: tuple[CellType, CellType, CellType, CellType]
    x_chips: int
    o_chips: int
    advantage: int
