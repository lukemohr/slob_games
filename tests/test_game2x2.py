"""Tests for game2x2.py."""

import numpy as np
import pytest

from constants import P1, P2
from game2x2 import SLOBGame2x2
from state import State


@pytest.fixture
def game():
    """Create simple symmetric 2x2 probabilities."""
    board_probs = np.array(
        [
            [0.4, 0.2, 0.4],
            [0.4, 0.2, 0.4],
            [0.4, 0.2, 0.4],
            [0.4, 0.2, 0.4],
        ]
    )
    g = SLOBGame2x2(board_probs, total_chips=4)
    # Fill legal move cache as generate_all_states typically would do
    from transitions import legal_moves

    all_boards = [
        (a, b, c, d)
        for a in [None, 0, 1]
        for b in [None, 0, 1]
        for c in [None, 0, 1]
        for d in [None, 0, 1]
    ]
    for board in all_boards:
        g.legal_moves_cache[board] = legal_moves(board)
    return g


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
    return SLOBGame2x2(board_probs, 3)


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


def test_successor_cache_initially_empty(game):
    assert len(game.successor_cache) == 0


def test_successors_populate_cache(game):
    s = State((None, None, None, None), 2, 2, 0)

    succ1 = game.get_successors(s)
    assert len(game.successor_cache) == 1
    assert s in game.successor_cache
    canonical_s = game.canonical_map.get(s.board, s)
    assert canonical_s in game.successor_cache

    # Second call must reuse cache
    succ2 = game.get_successors(s)
    assert succ1 is succ2


def test_terminal_has_no_successors_and_not_cached(game):
    # Terminal example: X wins on first row
    terminal_board = (0, 0, None, None)
    s = State(terminal_board, 3, 1, 0)

    succ = game.get_successors(s)
    assert succ == {}

    canonical_s = game.canonical_map[terminal_board]
    # Terminal states should NOT be cached
    assert canonical_s not in game.successor_cache


def test_successor_probabilities_sum_to_one(game):
    s = State((None, None, None, None), 2, 2, 0)

    succ = game.get_successors(s)
    total_prob = sum(succ.values())

    assert abs(total_prob - 1.0) < 1e-12


def test_successors_are_canonical_states(game):
    s = State((None, None, None, None), 2, 2, 0)
    succ = game.get_successors(s)

    for next_state in succ:
        # If next_state is terminal, it must equal canonical_map entry
        term, _ = game.is_terminal(next_state.board)
        if term:
            assert next_state is game.canonical_map[next_state.board]
