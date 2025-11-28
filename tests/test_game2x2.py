"""Tests for game2x2.py."""

import numpy as np

from constants import P1, P2
from game2x2 import SLOBGame2x2
from state import State


def example_game():
    """Create an example 2x2 game."""
    board_probs = np.array(
        [
            [1.0, 0.0, 0.0],  # always good
            [0.5, 0.25, 0.25],  # always neutral
            [0.3, 0.4, 0.3],  # always bad
            [0.8, 0.1, 0.1],
        ]
    )
    return SLOBGame2x2(board_probs)


def test_terminal():
    """Tests for is_terminal."""
    # X wins horizontally
    game = example_game()
    assert game.is_terminal((P1, P1, None, None)) == (True, P1)

    # O wins vertically
    assert game.is_terminal((P2, None, P2, None)) == (True, P2)

    # draw
    assert game.is_terminal((P1, P2, P2, P1)) == (True, 0.5)

    # not terminal
    assert game.is_terminal((None, P1, P2, None)) == (False, None)


def test_successors():
    """Test successors."""
    game = example_game()
    s = State((None, None, None, None), 1, 1, P1)
    succ = game.successors(s)

    # Should produce several states
    assert len(succ) > 0
    # Probabilities sum to 1
    assert abs(sum(succ.values()) - 1.0) < 1e-6
