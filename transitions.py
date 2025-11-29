"""Functions for state transitions."""

from typing import Any

from state import State


def canonical_terminal_state(board):
    """Create the canonical terminal state from the board."""
    return State(board, x_chips=0, o_chips=0, advantage=0)


def check_winner(win_lines, board) -> tuple[bool, int | None]:
    """Check if there is a winner on the board.

    Args:
        win_lines (list): The win conditions for the game.
        board (tuple): Current board state.

    Returns:
        tuple[bool, int | None]: If the game is won, who the winner is or None if no
        winner yet.

    """
    # Check X wins
    for a, b in win_lines:
        if board[a] == 0 and board[b] == 0:
            return True, 0

    # Check O wins
    for a, b in win_lines:
        if board[a] == 1 and board[b] == 1:
            return True, 1

    return False, None


def legal_moves(board) -> list[int]:
    """Get all legal moves for the current board state.

    Args:
        board (tuple): The current state of the board.

    Returns:
        list: A list of indices representing legal moves.

    """
    return [i for i, x in enumerate(board) if x is None]


def apply_mark(board, cell, player) -> tuple:
    """Update the board based on player and selected cell.

    Args:
        board (tuple): The current state of the board.
        cell (int): The cell on teh board to be updated.
        player (int): The player whose mark is being placed.

    Returns:
        tuple: The updated board.

    """
    b2 = list(board)
    b2[cell] = player
    return tuple(b2)


def apply_cell(board_probs, board, cell, player) -> list[tuple[tuple[Any, ...], Any]]:
    """Apply a move to the board and return possible outcomes.

    Args:
        board_probs (nparray): The probabilities on the game board.
        board (tuple): The current state of the board.
        cell (int): The cell to apply the move to.
        player (int): The player making the move (0 or 1).

    Returns:
        list: A list of tuples representing the next board state and its
        probability.

    """
    p_plus, p_zero, p_minus = board_probs[cell]

    results = []

    # + outcome
    if p_plus > 0:
        b2 = list(board)
        b2[cell] = player
        results.append((tuple(b2), p_plus))

    # 0 outcome
    if p_zero > 0:
        # board unchanged
        results.append((board, p_zero))

    # - outcome
    if p_minus > 0:
        b2 = list(board)
        b2[cell] = 1 - player
        results.append((tuple(b2), p_minus))

    return results


def is_full_board(board) -> bool:
    """Check if the board is full.

    Args:
        board (tuple): The current board state.

    Returns:
        bool: True if the board is full, False if spaces are available.

    """
    return all(cell is not None for cell in board)
