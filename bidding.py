"""Functions related to bidding."""

from constants import P1, P2
from state import State


def winner_of_bid(bx, bo, advantage) -> int:
    """Return the winner of a bid.

    Args:
        bx (int): Player 1's bid.
        bo (int): Player 2's bid.
        advantage (int): The player that currently holds the advantage token in bidding.

    Returns:
        int: The player who won the bid. 0 if P1, 1 if P2.

    """
    if bx > bo:
        return P1
    if bo > bx:
        return P2
    return advantage  # tie


def updated_chips_after_bid(Bx, Bo, bx, bo, winner) -> tuple[int, int]:
    """Update player chip holdings after bidding.

    Args:
        Bx (int): Player 1's current chip count.
        Bo (int): Player 2's current chip count.
        bx (int): Player 1's bid.
        bo (int): Player 2's bid.
        winner (int): The winner of the bid.

    Returns:
        tuple[int, int]: Player 1's updated chip count, Player 2's updated chip count.

    """
    if winner == P1:
        return Bx - bx, Bo + bx
    else:
        return Bx + bo, Bo - bo


def next_advantage(bx, bo, advantage) -> int:
    """Update the advantage if a tie occurs.

    Args:
        bx (int): Player 1's bet.
        bo (int): Player 2's bet.
        advantage (int): The current player holding the advantage token.

    Returns:
        int: The new player that holds the advantage.

    """
    if bx == bo:
        return 1 - advantage
    return advantage


def bidding_outcomes(state: State) -> list[tuple[int, int, int, int, int, int]]:
    """Generate all possible bidding outcomes for the current state.

    Args:
        state (State): The current game state.

    Returns:
        list: A list of tuples representing bidding outcomes.

    """
    Bx, Bo = state.x_chips, state.o_chips
    adv = state.advantage

    outcomes = []

    for bx in range(Bx + 1):
        for bo in range(Bo + 1):
            # Determine winner
            winner = winner_of_bid(bx, bo, adv)

            # Chip transfers ("winner pays", Richman bidding)
            next_x, next_o = updated_chips_after_bid(Bx, Bo, bx, bo, winner)

            # Advantage only flips on tie
            next_adv = next_advantage(bx, bo, adv)

            outcomes.append((bx, bo, winner, next_x, next_o, next_adv))

    return outcomes
