"""Functions for dynamic programming solver."""

from itertools import product
from typing import Sequence

import numpy as np
from scipy.optimize import linprog  # type: ignore

from bidding import bidding_outcomes
from constants import P1, P2
from game2x2 import SLOBGame2x2
from state import CellType, State, canonicalize_state
from transitions import apply_cell, legal_moves


def count_wins_for_player(
    win_lines: list[tuple[int, ...]],
    board: Sequence[CellType],
    player: int,
) -> int:
    """Count how many winning lines 'player' has on this board.

    Args:
        win_lines: Win conditions for the game.
        board: Current board state.
        player: Player to examine.

    Returns:
        int: Number of win conditions on the board for the player.

    """
    wins = 0
    for line in win_lines:
        # all cells in the line must equal `player`
        if all(board[i] == player for i in line):
            wins += 1
    return wins


def is_legal_board(
    win_lines: list[tuple[int, ...]],
    board: tuple,
) -> bool:
    """Check if board is achievable based on the win conditions.

    Rules:
      - Illegal if both players have at least one winning line.
      - If neither player has a winning line: legal.
      - If exactly one player has winning line(s), then there must exist
        a 'last move' cell belonging to that player such that if we set
        it back to None, the resulting board has no wins for either player.

    Args:
        win_lines: Win conditions for the game.
        board: Current board state.

    Returns:
        bool: True if board is achievable, False otherwise.

    """
    x_wins = count_wins_for_player(win_lines, board, P1)
    o_wins = count_wins_for_player(win_lines, board, P2)

    # Case 1: both players win -> impossible
    if x_wins > 0 and o_wins > 0:
        return False

    # Case 2: no one wins -> always legal (could be mid-game or draw candidate)
    if x_wins == 0 and o_wins == 0:
        return True

    # Case 3: exactly one player wins
    winner = P1 if x_wins > 0 else P2

    # Try to find a plausible 'last move' cell
    for i, cell in enumerate(board):
        if cell == winner:
            # Undo this move
            pre_board = list(board)
            pre_board[i] = None
            pre_board = tuple(pre_board)  # type: ignore

            # Recount wins on the 'previous' board
            pre_x_wins = count_wins_for_player(win_lines, pre_board, 0)
            pre_o_wins = count_wins_for_player(win_lines, pre_board, 1)

            # Previous board must have no wins at all
            if pre_x_wins == 0 and pre_o_wins == 0:
                return True

    # If no candidate last-move cell works, board is unreachable
    return False


def generate_all_states(game: SLOBGame2x2) -> list[State]:
    """Generate all possible states of the game.

    This includes:
        - all legal board configurations
        - X and O chip counts in [0, max_chips]
        - advantage ∈ {X, O}

    Args:
        game: The game being played.

    Returns:
        list[State]: All possible states in the game.

    """
    # TODO: This is hardcoded for 2x2 game right now. we will want to extend.
    all_boards = list(product([None, 0, 1], repeat=4))
    legal_boards = [
        board for board in all_boards if is_legal_board(game.win_lines, board)
    ]
    game.legal_moves_cache = {board: legal_moves(board) for board in legal_boards}
    chip_distributions = [
        (game.total_chips - i, i) for i in range(game.total_chips + 1)
    ]
    advantages = [0, 1]
    states = []
    for board in legal_boards:
        is_term, _ = game.is_terminal(board)

        if is_term:
            # Terminal: keep ONE canonical version
            # (chips and advantage are irrelevant)
            states.append(game.canonical_map[board])
            continue

        # Non-terminal: enumerate all chip-distribution × advantage combinations
        for (x, o), adv in product(chip_distributions, advantages):
            states.append(
                State(
                    board=board,
                    x_chips=x,
                    o_chips=o,
                    advantage=adv,
                )
            )

    return list(set(states))


def initialize_values(states: list[State], game: SLOBGame2x2) -> dict[State, float]:
    """Initialize the value function V.

    The initial values are:
        - 1.0 for X terminal wins
        - 0.0 for O terminal wins
        - 0.5 for draws
        - 0.5 for all other states as starting point

    Args:
        states (list): List of all possible states in the game.
        game (Game): The game being played.

    Returns:
        dict: Initial values for the possible states in the game.

    """
    initial_values = {}
    terminal_boards = game.find_terminal_boards()

    for state in states:
        if state.board in terminal_boards:
            initial_values[state] = terminal_boards[state.board]
        else:
            initial_values[state] = 0.5
    return initial_values


def expected_value_of_move(
    board: tuple,
    cell: int,
    winner: int,
    nx: int,
    no: int,
    next_adv: int,
    game: SLOBGame2x2,
    V: dict[State, float],
) -> float:
    """Compute the expected value of bid winner selecting a given cell.

    Args:
        board: current game board
        cell: cell index chosen by the player who won the bid
        winner: X or O (0 or 1)
        nx: next X chip counts after the bidding outcome
        no: next O chip counts after the bidding outcome
        next_adv: next advantage after the bidding outcome
        game: the SLOBGame instance (provides board_probs)
        V: value function table (maps State -> float)

    Returns:
        float: expected value of this move under current V

    """
    exp_value = 0.0

    # Stochastic outcomes of applying this move
    for next_board, p_outcome in apply_cell(game.board_probs, board, cell, winner):
        next_state = State(
            board=next_board,
            x_chips=nx,
            o_chips=no,
            advantage=next_adv,
        )
        next_state = canonicalize_state(next_state, game.canonical_map)
        exp_value += p_outcome * V[next_state]

    return exp_value


def compute_move_values_for_winner(
    board: tuple,
    winner: int,
    nx: int,
    no: int,
    next_adv: int,
    game: SLOBGame2x2,
    V: dict[State, float],
    moves: list[int],
) -> list[float]:
    """Compute expected value of all legal moves for the specified winner.

    Args:
        board: current game board
        winner: X or O (0 or 1)
        nx: next X chip counts after bid outcome
        no: next O chip counts after bid outcome
        next_adv: next advantage after a tie outcome
        game: the game instance
        V: value table
        moves: the moves to be evaluated

    Returns:
        (legal_moves, move_values):
            - legal_moves: list of cell indices
            - move_values: list of expected values for each cell

    """
    values = []

    for cell in moves:
        ev = expected_value_of_move(board, cell, winner, nx, no, next_adv, game, V)
        values.append(ev)

    return values


def build_bidding_matrix(
    state: State,
    game: SLOBGame2x2,
    V: dict[State, float],
) -> np.ndarray:
    """Build the bidding payoff matrix M for a given state.

    For each possible (bx, bo):
        - Determine the winner
        - Compute optimal move value for that winner:
            X --> max over legal moves
            O --> min over legal moves
        - Insert into M[bx][bo]

    Returns:
        M: a matrix of shape [(x_chips+1), (o_chips+1)]

    """
    M_s = np.zeros((state.x_chips + 1, state.o_chips + 1))
    legal = game.legal_moves_cache[state.board]

    for bx, bo, winner, nx, no, next_adv in bidding_outcomes(state):
        # Compute all move EVs (list of float)
        move_vals = compute_move_values_for_winner(
            board=state.board,
            winner=winner,
            nx=nx,
            no=no,
            next_adv=next_adv,
            game=game,
            V=V,
            moves=legal,
        )

        # Optimal expected value based on which player won the bid
        if winner == P1:  # or X
            optimal_value = max(move_vals) if move_vals else 0.5
        else:
            optimal_value = min(move_vals) if move_vals else 0.5

        M_s[bx, bo] = optimal_value

    return M_s


def solve_zero_sum_matrix_game(M: np.ndarray) -> float:
    """Solve the zero-sum matrix game defined by payoff matrix M.

    Computes:
        V = max_p min_q p^T M q

    Using standard LP formulation for the maximizing player (X).

    Variables:
        p_i for i in rows of M (probabilities for X)
        V   scalar game value

    Maximize V  <=>  minimize -V

    Constraints:
        sum_i p_i = 1
        p_i >= 0
        For each column j:
            sum_i p_i * M[i,j] >= V
        Rearranged for linprog (Ax <= b):
            -sum_i p_i*M[i,j] + V <= 0

    Args:
        M: np.ndarray of shape (R, C)

    Returns:
        float: game value V in [min(M), max(M)]

    """
    M = np.array(M, dtype=float)
    R, C = M.shape

    # -----------------------------------------------------
    # Variables:
    #   x = [p_0, p_1, ..., p_{R-1}, V]
    # Dimension = R + 1
    # -----------------------------------------------------
    num_vars = R + 1
    idx_V = R  # last variable is V

    # -----------------------------------------------------
    # Objective: maximize V  <=> minimize -V
    # c = [0, ..., 0, -1]
    # -----------------------------------------------------
    c = np.zeros(num_vars)
    c[idx_V] = -1.0

    # -----------------------------------------------------
    # Equality constraint: sum_i p_i = 1
    #
    # A_eq: shape (1, R+1)
    # A_eq[0,0:R] = 1,  A_eq[0,R] = 0
    # -----------------------------------------------------
    A_eq = np.zeros((1, num_vars))
    A_eq[0, :R] = 1.0
    b_eq = np.array([1.0])

    # -----------------------------------------------------
    # Inequality constraints:
    #
    # For each column j:
    #     sum_i p_i*M[i,j] >= V
    #
    # Rearrange:
    #    -sum_i p_i*M[i,j] + V <= 0
    #
    # A_ub[j,0:R] = -M[:,j]
    # A_ub[j,R]   = 1
    #
    # b_ub[j] = 0
    # -----------------------------------------------------
    A_ub = np.zeros((C, num_vars))
    b_ub = np.zeros(C)

    for j in range(C):
        A_ub[j, :R] = -M[:, j]  # -sum_i p_i*M[i,j]
        A_ub[j, idx_V] = 1.0  # + V

    # -----------------------------------------------------
    # Bounds:
    #   p_i >= 0
    #   V free (but we can bound V for numerical stability)
    #
    # Because M's payoffs are always in [0,1], we can safely
    # bound V to [-1, +1] to help the LP solver.
    # -----------------------------------------------------
    bounds = [(0.0, 1.0) for _ in range(R)] + [(-1.0, 1.0)]

    # -----------------------------------------------------
    # Solve LP
    # -----------------------------------------------------
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        # Degeneracy fallback: slightly perturb M and retry
        M_perturbed = M + 1e-8 * np.random.randn(*M.shape)

        A_ub_perturbed = np.zeros_like(A_ub)
        for j in range(C):
            A_ub_perturbed[j, :R] = -M_perturbed[:, j]
            A_ub_perturbed[j, idx_V] = 1.0

        result2 = linprog(
            c,
            A_ub=A_ub_perturbed,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not result2.success:
            raise RuntimeError(
                f"Zero-sum LP failed to solve even after perturbation: {result.message}"
            )

        return float(result2.x[idx_V])

    # V is the last variable
    return float(result.x[idx_V])


def value_iteration(
    states: list[State],
    game: SLOBGame2x2,
    V_init: dict[State, float],
    tolerance: float = 1e-6,
    max_iters: int = 1000,
) -> dict[State, float]:
    """Run value iteration until convergence.

    Args:
        states: All states being evaluated.
        game: Game being played.
        V_init: Initial value for V.
        tolerance: Tolerance value for convergence.
        max_iters: Maximum iterations in attempting convergence.

    Returns:
        Final V after iterations.

    """
    v = V_init.copy()

    for i in range(max_iters):
        v_new = {}
        max_diff = 0.0
        for state in states:
            if game.is_terminal(state.board)[0]:
                v_new[state] = v[state]
                continue
            M_s = build_bidding_matrix(state, game, v)
            v_new[state] = solve_zero_sum_matrix_game(M_s)
            diff = abs(v_new[state] - v[state])
            if diff > max_diff:
                max_diff = diff
        if max_diff < tolerance:
            print(f"Converged after {i} iterations.")
            return v_new
        v = v_new

    print(f"Warning: V did not converge in {max_iters} iterations")
    return v
