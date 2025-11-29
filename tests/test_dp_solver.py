"""Tests for dp_solver.py."""

from collections import defaultdict

import numpy as np
import pytest

from dp_solver import (
    build_bidding_matrix,
    generate_all_states,
    initialize_values,
    is_legal_board,
    solve_zero_sum_matrix_game,
    value_iteration,
)
from state import State

WIN_LINES_2x2 = [(0, 1), (2, 3), (0, 2), (1, 3)]


@pytest.mark.parametrize(
    "board",
    [
        (None, None, None, None),
        (0, None, None, None),
        (0, 1, None, None),
        (0, 0, None, None),  # X wins
        (1, None, 1, None),  # O wins
    ],
)
def test_board_legal(board):
    """Test is_legal_board."""
    assert is_legal_board(WIN_LINES_2x2, board)


def test_both_players_win_illegal():
    """Test illegal board with two winners."""
    # X wins on top row, O wins on bottom row
    board = (0, 0, 1, 1)
    assert not is_legal_board(WIN_LINES_2x2, board)


def test_two_win_lines_for_same_player_must_be_reachable():
    """Test edge board."""
    # X wins on both (0,1) and (0,2) but achievable:
    # Last move at cell 1 gives wins at 0-1 and 0-2 simultaneously
    board = (0, 0, 0, None)
    assert is_legal_board(WIN_LINES_2x2, board)


def test_solver_identity_matrix():
    """Test simple M solver."""
    M = np.eye(2)
    # Player X chooses row to maximize V = min column value.
    # Identity gives V = 1 if pick row 1, but min(M[row]) = 0 for row 0.
    # Optimal mixed = 100% row 1
    val = solve_zero_sum_matrix_game(M)
    assert abs(val - 0.5) < 1e-6


def test_solver_constant_matrix():
    """Test constant M."""
    M = np.full((3, 4), 0.3)
    val = solve_zero_sum_matrix_game(M)
    assert abs(val - 0.3) < 1e-6


def test_solver_simple_bimatrix():
    """Test offset diagonal M."""
    M = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    # Classic matching pennies → value = 0.5
    val = solve_zero_sum_matrix_game(M)
    assert abs(val - 0.5) < 1e-6


def test_bidding_matrix_single_move_choice(game2x2):
    """Test simple M."""
    board = (None, None, None, None)
    state = State(board, 1, 1, 0)

    # V[next_state] is always 0.7, for any state
    V = defaultdict(lambda: 0.7)

    M = build_bidding_matrix(state, game2x2, V)

    # All entries of M should be 0.7
    assert np.allclose(M, 0.7)


def test_value_iteration_converges_trivial(game2x2):
    """Test value iteration convergence."""
    states = generate_all_states(game2x2, total_chips=2)
    V_init = initialize_values(states, game2x2)

    V_final = value_iteration(states, game2x2, V_init, tolerance=1e-6, max_iters=50)

    # Sanity checks
    assert isinstance(V_final, dict)
    assert len(V_final) == len(states)
    assert all(0.0 <= val <= 1.0 for val in V_final.values())
