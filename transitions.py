"""Functions for state transitions."""

from typing import Any


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
